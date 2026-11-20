"""Bridge for the settings panel INSIDE the VRChat world.

A world can only send GET requests to fixed addresses, but it can write lines to VRChat's output_log,
which the app already reads. So:

    world -> app   log lines   "[W04AI] CFG_OPEN"         the panel was opened / refreshed
                               "[W04AI] CFG_SET {json}"   one edit (see OPS below)
    app -> world   GET /cfg    the current settings as compact JSON (no token)

Because any world may GET 127.0.0.1 and may write log lines, both directions are limited:
  * /cfg answers only for WINDOW_S seconds after a CFG_OPEN line;
  * CFG_* lines count only when they come from a log file whose current world is allowed: a local
    test world (Build & Test, "local:...") or an ID in the combined W04_WORLD_IDS allowlist.
persona_npc_app.py tracks the world of each log file and calls handle_line() only for new lines, so
lines written before the app started are never applied again.

Known limit: Udon's Debug.Log lines look exactly like VRChat's own lines, so a world written on purpose
to attack this app could also fake a "Joining ..." line. The world check stops ordinary worlds, not that.
"""
import hashlib
import json
import re
import threading
import time

import persona_store

WINDOW_S = 20.0            # /cfg answers this long after CFG_OPEN
DICTATE_S = 60.0           # an armed dictation waits this long for the next push-to-talk
OPEN_GRACE_S = 2.0         # /cfg may arrive a moment before the CFG_OPEN line is flushed to the log
MAX_CFG_BYTES = 16000      # whole /cfg answer, UTF-8
HISTORY_ITEMS = 10
DICTATE_TARGETS = ("persona_append", "note_add", "impression")

_STAMP = r"^\d{4}\.\d\d\.\d\d \d\d:\d\d:\d\d \w+\s+-\s+"     # every real log line starts like this
CFG_RE = re.compile(_STAMP + r"\[W04AI\] (CFG_OPEN|CFG_SET)\b(.*)$", re.DOTALL)
# 2026.11.19 13:44:37 Debug      -  [Behaviour] Joining local:1a7a5121...
# 2026.11.19 13:44:06 Debug      -  [Behaviour] Joining wrld_b05ed97c-...:59823~private(usr_...)~region(jp)
# ("[Behaviour] Joining or Creating Room: <name>" is a different line and does not match)
JOIN_RE = re.compile(_STAMP + r"\[Behaviour\] Joining (local:[0-9A-Za-z]+|wrld_[0-9A-Fa-f\-]+)")
LEAVE_RE = re.compile(_STAMP + r"(\[Behaviour\] OnLeftRoom\b|VRCApplication: HandleApplicationQuit\b)")


def parse_world_ids(text: str) -> set:
    """'wrld_a, wrld_b' (W04_WORLD_IDS) -> {'wrld_a', 'wrld_b'}; anything that is not a world ID is dropped."""
    return {w for w in (p.strip() for p in (text or "").split(",")) if re.fullmatch(r"wrld_[0-9A-Fa-f\-]+", w)}


def load_world_ids(configured_ids: str, example_path) -> set:
    """Combine the effective environment setting with the bundled world IDs."""
    world_ids = parse_world_ids(configured_ids)
    try:
        with open(example_path, encoding="utf-8-sig") as example:
            for raw in example:
                key, separator, value = raw.partition("=")
                if separator and key.strip() == "W04_WORLD_IDS":
                    world_ids.update(parse_world_ids(value.strip().strip('"').strip("'")))
                    break
    except FileNotFoundError:
        pass
    return world_ids


def world_from_line(line: str, current):
    """World of a log file after this line: 'local:...' / 'wrld_...' / None (left)."""
    if "Joining " in line:
        m = JOIN_RE.match(line)
        if m:
            return m.group(1)
    elif "OnLeftRoom" in line or "HandleApplicationQuit" in line:
        if LEAVE_RE.match(line):
            return None
    return current


def _flag(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "on", "yes")
    return bool(v)


def _index(v):
    """VRCJson writes numbers as doubles: 2.0 -> 2. Anything else that is not a whole number -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v if isinstance(v, int) else None


class CfgBridge:
    def __init__(self, store: "persona_store.Store", world_ids=(), log=None, clock=time.monotonic):
        self.store = store
        self._bridges = {store.npc: self}
        self._shared = {"dictate": None, "npc": "kwonjun"}
        self.world_ids = set(world_ids)
        self.log = log or (lambda kind, detail="", say="": None)
        self.clock = clock
        self.cond = threading.Condition()
        self.open_until = 0.0
        self.open_source = None            # the log file whose panel opened the window
        self.dictate = None                # {"target", "player", "until", "source"}
        self.msg = ""                      # result of the last edit, shown by the panel
        self.rev = int(time.time()) - 1_700_000_000    # keeps growing across restarts
        self._digest = None
        self.persona_cut = False           # the last /cfg answer carried only part of the persona
        self.notes_cut = False             # ... or only part of the memo list

    @property
    def dictate(self):
        return self._shared["dictate"]

    @dictate.setter
    def dictate(self, value):
        self._shared["dictate"] = value

    def for_npc(self, npc=None):
        selected = persona_store.npc_id(npc)
        with self.cond:
            if selected not in self._bridges:
                child = CfgBridge(self.store.for_npc(selected), self.world_ids, self.log, self.clock)
                child._bridges, child._shared, child.cond = self._bridges, self._shared, self.cond
                self._bridges[selected] = child
            return self._bridges[selected]

    # ------------------------------------------------------------ gates

    def world_allowed(self, world) -> bool:
        return bool(world) and (world.startswith("local:") or world in self.world_ids)

    def world_changed(self, source) -> None:
        """The client writing `source` joined or left a world: its window and dictation end."""
        with self.cond:
            if self.open_source == source:
                self.open_until, self.open_source = 0.0, None
            if self.dictate and self.dictate["source"] == source:
                self.dictate = None

    def is_open(self) -> bool:
        with self.cond:
            return self.clock() < self.open_until

    def wait_open(self, timeout: float = None) -> bool:
        with self.cond:
            timeout = OPEN_GRACE_S if timeout is None else timeout
            return self.cond.wait_for(lambda: self.clock() < self.open_until, timeout=max(0.0, timeout))

    # ------------------------------------------------------------ world -> app

    def handle_line(self, source, world, line: str) -> bool:
        """One NEW log line. True if it was a CFG_* line (applied or refused)."""
        if "CFG_" not in line:
            return False
        m = CFG_RE.match(line)
        if not m:
            return False
        if not self.world_allowed(world):
            self.log("CFG_IGNORED", f"{m.group(1)}\tworld_not_allowed",
                     say="월드 설정 창 요청을 무시함 (허용하지 않은 월드)")
            return True
        if m.group(1) == "CFG_OPEN":
            with self.cond:
                selected = re.search(r"\bnpc=([^\s]+)", m.group(2))
                self._shared["npc"] = persona_store.npc_id(selected.group(1) if selected else None)
                self.open_until, self.open_source = self.clock() + WINDOW_S, source
                self.cond.notify_all()
            self.log("CFG_OPEN", f"window_s={WINDOW_S:.0f}", say=f"월드 설정 창이 열림 ({WINDOW_S:.0f}초 동안 내용을 보냄)")
            return True
        rest = m.group(2)
        try:
            req, _ = json.JSONDecoder().raw_decode(rest[rest.index("{"):])   # " utc=..." may follow the JSON
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            self.log("CFG_SET", "bad_json\trefused", say="월드 설정 창: 알 수 없는 요청이라 무시함")
            return True
        self.apply(req, source)
        return True

    def apply(self, req: dict, source=None) -> str:
        """One edit. Returns '' when applied, else the Korean reason (also kept in self.msg for the panel)."""
        selected = persona_store.npc_id(req.get("npc"))
        if selected != self.store.npc:
            return self.for_npc(selected).apply(req, source)
        op = req.get("op") if isinstance(req.get("op"), str) else ""
        text = req.get("text")
        n = len(text) if isinstance(text, str) else 0
        st = self.store
        player = req.get("player")
        if op in ("impression", "lock", "dictate") and player == self._panel_speaker(st.npc_name()):
            player = st.speaker
        what, detail = "", f"op={op[:20]}\tlen={n}"
        if op == "load":
            self._shared["npc"] = st.npc
            err, what = "", "load"
        elif op in ("persona", "name", "impression", "note_add", "note_edit", "notes_set") and not isinstance(text, str):
            err = "글이 없습니다."
        elif op == "persona" and self.persona_cut:
            err = "성격 글이 길어서 월드 창에는 일부만 보였습니다. 웹 설정 화면에서 고쳐 주세요."
        elif op == "persona":
            err, what = st.set_persona(text), f"성격 글을 고침 ({min(n, persona_store.MAX_PERSONA_CHARS)}자)"
        elif op == "name":
            err, what = st.set_npc_name(text), "NPC 이름을 고침"
        elif op == "impression":
            err, what = st.set_impression(player, text), f"플레이어 인상을 고침 ({min(n, persona_store.MAX_IMPRESSION_CHARS)}자)"
        elif op == "lock":
            on = _flag(req.get("on"))
            err, what = st.set_locked(player, on), "플레이어 인상을 " + ("잠금" if on else "잠금 풂")
            detail += f"\ton={int(on)}"
        elif op == "note_add":
            err, what = st.add_note(text), f"기억 메모를 더함 ({min(n, persona_store.MAX_NOTE_CHARS)}자)"
        elif op in ("note_edit", "note_del"):
            i = _index(req.get("index"))
            err = st.edit_note(i, text if op == "note_edit" else "", req.get("expect"))
            what = "기억 메모를 고침" if op == "note_edit" and text.strip() else "기억 메모를 지움"
            detail += f"\tindex={i}"
        elif op == "notes_set" and self.notes_cut:
            err = "메모가 많아서 월드 창에는 일부만 보였습니다. 웹 설정 화면에서 고쳐 주세요."
        elif op == "notes_set":
            # the panel's memo editor: one memo per line. Lines kept as they were keep their author/time; memos the
            # AI wrote while the player was editing (not in "expect", the lines the editor started from) are kept too.
            before = req.get("expect") if isinstance(req.get("expect"), str) else ""
            seen = {ln.strip() for ln in before.splitlines() if ln.strip()}
            old = {n["text"]: n for n in st.notes}
            lines = [ln.strip() for ln in text.replace("\r", "\n").splitlines() if ln.strip()]
            new = [old.get(ln, {"text": ln, "by": "user"}) for ln in lines]
            added = [n for n in st.notes if n["text"] not in seen and n["text"] not in lines]
            st.apply({"notes": new + added})
            err, what = "", f"기억 메모를 고침 ({len(st.notes)}개)"
            detail += f"\tlines={len(lines)}\tkept_new={len(added)}"
        elif op == "history_clear":
            st.apply({"history_clear": True})
            err, what = "", "최근 대화를 모두 잊음"
        elif op == "model":
            try:
                st.apply({"settings": {"brain_model": req.get("model")}})
                err, what = "", "다음 대답부터 쓸 모델을 고침"
            except ValueError as e:
                err = str(e)
        elif op == "smalltalk":
            try:
                st.set_smalltalk_settings({key: req[key] for key in persona_store.DEFAULT_SMALLTALK if key in req})
                err, what = "", "두 NPC의 잡담 설정을 고침"
            except (ValueError, OSError) as e:
                err = str(e) if isinstance(e, ValueError) else "잡담 설정을 저장하지 못했습니다."
        elif op == "effort":
            try:
                st.apply({"settings": {"brain_effort": req.get("effort")}})
                err, what = "", "다음 대답부터 쓸 생각 깊이를 고침"
            except ValueError as e:
                err = str(e)
        elif op in ("auto", "keep_history", "recall", "compact"):
            on = _flag(req.get("on"))
            key, label = {"auto": ("auto_update", "스스로 고쳐 쓰기"), "keep_history": ("keep_history", "최근 대화 이어가기"),
                          "recall": ("auto_recall", "필요할 때 깊이 떠올리기"),
                          "compact": ("compact_on", "기억 자동 정리")}[op]
            st.apply({"settings": {key: on}})
            err = ""
            what = label + (" 켬" if on else " 끔")
            detail += f"\ton={int(on)}"
        elif op == "compact_undo":
            err, what = st.compaction_undo(), "기억 정리를 되돌림 (자동 정리는 꺼 둠)"
        elif op == "dictate":
            target = req.get("target")
            if target == "none":
                with self.cond:
                    if self.dictate and self.dictate.get("npc", "kwonjun") == st.npc:
                        self.dictate = None
                err, what = "", "받아쓰기 취소"
            elif target not in DICTATE_TARGETS:
                err = "알 수 없는 받아쓰기 대상입니다."
            else:
                player = (persona_store._one_line(player, persona_store.MAX_NAME_CHARS) or st.speaker
                          if target == "impression" else "")
                with self.cond:
                    self.dictate = {"target": target, "player": player, "until": self.clock() + DICTATE_S,
                                    "source": source, "npc": st.npc}
                err, what = "", f"받아쓰기 대기: 다음에 점프를 누르고 한 말을 적음 ({DICTATE_S:.0f}초 안에)"
                detail += f"\ttarget={target}"
        else:
            err = "알 수 없는 요청입니다."
        self.msg = err
        self.log("CFG_SET", detail + ("\trefused" if err else "\tok"),
                 say="월드 설정 창: " + (f"적용 안 함 ({err})" if err else what))
        return err

    # ------------------------------------------------------------ dictation (next push-to-talk)

    def dictate_target(self):
        """The armed dictation, or None (also None once it expired)."""
        with self.cond:
            d = self.dictate
            if d and self.clock() >= d["until"]:
                self.dictate = d = None
            return dict(d) if d else None

    def dictate_apply(self, text: str, expected=None) -> str:
        """Put the transcript into the armed target and disarm. '' when stored, else the reason."""
        with self.cond:
            if expected is not None and self.dictate_target() != expected:
                return "받아쓰기 대상이 바뀌었거나 대기 시간이 끝났습니다."
            d, self.dictate = self.dictate, None
        if not d:
            return "받아쓰기 대기 중이 아닙니다."
        st = self.store.for_npc(d.get("npc"))
        if d["target"] == "persona_append":
            err = st.append_persona(text)
        elif d["target"] == "note_add":
            err = st.add_note(text)
        else:
            err = st.set_impression(d["player"], text)
        self.for_npc(d.get("npc")).msg = err or "받아쓴 말을 적었습니다."
        self.log("CFG_DICTATE", f"target={d['target']}\tlen={len(text)}" + ("\trefused" if err else "\tok"),
                 say="월드 설정 창: 받아쓰기 " + (f"적용 안 함 ({err})" if err else f"적음 ({d['target']}, {len(text)}자)"))
        return err

    # ------------------------------------------------------------ app -> world

    def _panel_speaker(self, npc_name: str) -> str:
        st = self.store
        return st.player_label(st.speaker) if st.speaker.casefold() == npc_name.casefold() else st.speaker

    def payload(self, npc=None) -> bytes:
        """The /cfg answer (the caller checks is_open() first)."""
        selected = persona_store.npc_id(npc if npc is not None else self._shared["npc"])
        return self.for_npc(selected)._payload()

    def _payload(self) -> bytes:
        st = self.store
        smalltalk = st.smalltalk_settings()
        d = self.dictate_target()
        if d and d.get("npc", "kwonjun") != st.npc:
            d = None
        with st.lock:
            npc_name = st.npc_name()
            speaker_label = self._panel_speaker(npc_name)
            players = [{"name": k, "note": v["note"], "locked": bool(v["locked"])} for k, v in st.players.items()]
            if st.speaker not in st.players:                    # so the panel can always edit "me"
                players.append({"name": st.speaker, "note": "", "locked": False})
            players.sort(key=lambda p: p["name"] != st.speaker)   # the speaker first
            players[0]["name"] = speaker_label
            history = []
            for m in st.history:
                said = m["content"]
                if m["role"] == "assistant":
                    try:
                        said = json.loads(said)["lines"]["ko"]
                    except (ValueError, KeyError, TypeError):
                        pass
                history.append({"who": "user" if m["role"] == "user" else "npc", "text": str(said)})
            body = {
                "ok": True, "rev": 0,
                "npc": npc_name, "speaker": speaker_label, "persona": st.persona_text(),
                "auto": bool(st.settings["auto_update"]), "keep_history": bool(st.settings["keep_history"]),
                "smalltalk": smalltalk,
                "players": players, "notes": [n["text"] for n in st.notes],
                "history": history[-HISTORY_ITEMS:],
                "dictate": {"target": d["target"], "player": d["player"]} if d else None,
                "msg": self.msg,
            }

        def pack() -> bytes:
            return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        with self.cond:
            body["rev"] = 2 ** 31                               # widest value while measuring
            while len(pack()) > MAX_CFG_BYTES and body["history"]:
                body["history"].pop(0)                          # oldest talk first
            while len(pack()) > MAX_CFG_BYTES and len(body["players"]) > 1:
                body["players"].pop()
            while len(pack()) > MAX_CFG_BYTES and body["persona"]:
                over = len(pack()) - MAX_CFG_BYTES              # bytes; a Korean letter is 3
                body["persona"] = body["persona"][:-max(1, over // 3)]
                body["persona_cut"] = True                      # the panel must not save this text back
            while len(pack()) > MAX_CFG_BYTES and body["notes"]:
                body["notes"].pop()                             # still too big (60 memos of 4-byte letters)
                body["notes_cut"] = True                        # the panel must not save this list back
            self.persona_cut = bool(body.get("persona_cut"))
            self.notes_cut = bool(body.get("notes_cut"))
            body["rev"] = 0
            digest = hashlib.sha256(pack()).hexdigest()
            if digest != self._digest:
                self._digest, self.rev = digest, self.rev + 1
            body["rev"] = self.rev
            return pack()
