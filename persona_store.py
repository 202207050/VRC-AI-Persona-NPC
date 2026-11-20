"""What the NPC keeps between replies: persona text, impressions of players, long-term notes, history.

Everything is a plain file so it can also be edited by hand:
    personas/<id>.md                 personality (PERSONA_FILE overrides kwonjun only)
    data/npcs/<id>/settings.json      switches from the settings page
    data/npcs/<id>/impressions.json   one short note per player
    data/npcs/<id>/memory.json        long-term memory notes
    data/npcs/<id>/history.json       recent conversation, when "keep history" is on
    data/npcs/<id>/compaction.json    automatic memory tidy-up: last result and the last 3 backups (undo)
    data/npcs/kwonjun/smalltalk_state.json  shared NPC small-talk settings and today's AI count
Legacy data files are backed up and copied to kwonjun on first open.
data/ is personal and is listed in .gitignore.

The settings page (persona_npc_app.py, /settings) and the AI (persona_ai.py) share one Store object;
the AI reads it on every reply, so edits apply to the next reply without restarting.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import time

from voice_options import (DEFAULT_EFFORT, DEFAULT_MODEL, MODEL_RATES, clean_hints, effort_options, model_options,
                           validate_effort, validate_model)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PERSONA = "persona.md"
NPC_IDS = ("kwonjun", "itoa")
DEFAULT_NPC_NAMES = {"kwonjun": "kwonjun", "itoa": "ITOAR"}
LEGACY_NPC_NAMES = {"kwonjun": "권준", "itoa": "이토아"}
DEFAULT_SMALLTALK = {"ai": True, "ratio": 0.2, "daily_cap": 30}
SMALLTALK_STATE = "smalltalk_state.json"


def npc_id(value=None):
    """Only known IDs select files; missing/unknown IDs use kwonjun."""
    return value if isinstance(value, str) and value in NPC_IDS else "kwonjun"


# "[W04AI] AFFECTION type=<pat|hug|wave> npc=<id> by=<name>" -> a phrase added to that player's impression
AFFECTION_PHRASES = {"pat": "내 머리를 쓰다듬어 준 적이 있음.", "hug": "나를 안아 준 적이 있음.",
                     "wave": "나에게 손을 흔들어 인사해 준 적이 있음."}
AFFECTION_INTERVAL_S = 600        # at most one note per player per type in this long
TOUCH_AFFECTION = ("pat", "hug")  # each is recorded at most once per saved impression

DEFAULT_SPEAKER = "플레이어"     # used until the VRChat log tells us the local player's name

MAX_HISTORY = 10                  # remembered exchanges (user + NPC = one exchange)
MAX_PERSONA_CHARS = 6000          # the persona is sent with every reply: keep it affordable
MAX_IMPRESSION_CHARS = 500        # hand-written impressions; long ones are condensed by compaction
AI_IMPRESSION_CHARS = 150         # what the AI itself may write in a reply (keeps replies short and cheap)
MAX_NOTE_CHARS = 80
MAX_NOTES = 60                    # hard cap; compaction normally runs before it (default: more than 50)
MAX_PLAYERS = 50
MAX_NAME_CHARS = 40

DEFAULT_SETTINGS = {
    "keep_history": False,        # False = today's behaviour: the conversation is forgotten on restart
    "auto_update": True,          # the AI may rewrite impressions / add memory notes by itself
    "brain_model": DEFAULT_MODEL,
    "brain_effort": DEFAULT_EFFORT,   # output_config.effort of the normal reply (voice_options.EFFORTS)
    "auto_recall": True,              # the AI may ask for a second, slower look through all its memos
    "filter_overheard": True,
    "stt_hints": "",
    # automatic memory compaction (user 2026-11-19): one cheap Haiku call merges/condenses the memos
    "compact_on": True,
    "compact_note_count": 50,         # more memos than this ...
    "compact_note_chars": 3000,       # ... or more characters in all memos together ...
    "compact_impression_chars": 400,  # ... or one player's impression longer than this starts it
    "compact_user_notes": False,      # False = memos the user wrote are kept word for word
    "compact_minutes": 10,            # at most one compaction per NPC in this many minutes
}
COMPACT_FLAGS = ("compact_on", "compact_user_notes")
FLAG_KEYS = ("keep_history", "auto_update", "auto_recall", "filter_overheard") + COMPACT_FLAGS
VALIDATED_KEYS = (("brain_model", validate_model), ("brain_effort", validate_effort), ("stt_hints", clean_hints))

# Which memos a normal reply sees while "auto_recall" is on (the recall call always sees all of them).
PICK_ALL_UNDER = 12               # this many memos or fewer: all of them, as before
PICK_RELEVANT = 8                 # memos that share words with what was just said
PICK_RECENT = 4                   # the newest memos
PICK_USER = 4                     # the newest memos the user wrote by hand
_PICK_STRIP = re.compile(r"[\W_]+")


def _bigrams(text) -> set:
    """Word pieces for a cheap match that works for Korean without a dictionary: every two neighbouring
    letters inside a word, plus the first letter of one- and two-letter words ("귤", "귤을" -> "귤")."""
    out = set()
    for word in _PICK_STRIP.split(str(text or "").lower()):
        out.update(word[i:i + 2] for i in range(len(word) - 1))
        if 1 <= len(word) <= 2:
            out.add(word[0])
    return out
COMPACT_RANGES = {                    # whole numbers, inclusive
    "compact_note_count": (5, MAX_NOTES - 1),
    "compact_note_chars": (200, MAX_NOTES * MAX_NOTE_CHARS),
    "compact_impression_chars": (100, MAX_IMPRESSION_CHARS - 1),
    "compact_minutes": (1, 1440),
}
COMPACT_LABELS = {"compact_note_count": "메모 개수 기준", "compact_note_chars": "메모 글자 수 기준",
                  "compact_impression_chars": "인상 글자 수 기준", "compact_minutes": "정리 간격(분)"}
COMPACT_FILE = "compaction.json"
COMPACT_BACKUPS = 3
COMPACT_MAX_IMPRESSIONS = 5           # impressions condensed in one call (the longest first)

_NAME_RE = re.compile(r"^[ \t]*[-*]?[ \t]*이름[ \t]*[:：][ \t]*(.+?)[ \t]*$", re.MULTILINE)


def _one_line(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit]


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _compact_number(key: str, v) -> int:
    if type(v) is float and v.is_integer():
        v = int(v)  # JSON may carry whole numbers as doubles
    low, high = COMPACT_RANGES[key]
    if type(v) is not int or not low <= v <= high:
        raise ValueError(f"{COMPACT_LABELS[key]}: {low:,}부터 {high:,} 사이의 정수로 적어 주세요.")
    return v


def _chars(notes) -> int:
    return sum(len(n["text"]) for n in notes)


def _smalltalk_settings(patch: dict) -> dict:
    if not isinstance(patch, dict) or not patch or patch.keys() - DEFAULT_SMALLTALK.keys():
        raise ValueError("잡담 설정 항목을 확인해 주세요.")
    if "ai" in patch and type(patch["ai"]) is not bool:
        raise ValueError("AI 잡담의 켜기/끄기 값을 확인해 주세요.")
    if "ratio" in patch:
        v = patch["ratio"]
        if type(v) not in (int, float) or not 0 <= v <= 1:
            raise ValueError("AI 비율은 0부터 1 사이여야 합니다.")
    if "daily_cap" in patch:
        v = patch["daily_cap"]
        if type(v) is float and v.is_integer():
            v = int(v)  # VRCJson can send whole numbers as doubles
        if type(v) is not int or v < 0:
            raise ValueError("AI 일일 한도는 0 이상의 정수여야 합니다.")
        patch = dict(patch, daily_cap=v)
    return patch


class Store:
    def __init__(self, data_dir: str = None, persona_path: str = None, npc: str = "kwonjun",
                 default_model: str = DEFAULT_MODEL, *, migrate_persona: bool = True):
        self.lock = threading.RLock()
        self.npc = npc_id(npc)
        self.root_dir = data_dir or os.path.join(HERE, "data")
        self.data_dir = os.path.join(self.root_dir, "npcs", self.npc)
        self._stores = {self.npc: self}
        self._blocked = set()                 # files that could not be read or backed up: never overwritten
        self.load_problems = []               # [(file name, backup name or '')], reported once at startup
        self._registry_lock = threading.RLock()
        if self.npc == "kwonjun":
            self._migrate_data()
        default = (os.path.join("personas", self.npc + ".md")
                   if os.path.isdir(os.path.join(HERE, "personas")) or self.npc != "kwonjun"
                   else DEFAULT_PERSONA)
        persona = persona_path or (os.environ.get("PERSONA_FILE") if self.npc == "kwonjun" else None) or default
        self.persona_path = persona if os.path.isabs(persona) else os.path.join(HERE, persona)
        if migrate_persona:
            self._migrate_persona_names()
        self.speaker = DEFAULT_SPEAKER
        self.speaker_known = False            # True once the VRChat log gave a real display name
        self.default_model = validate_model(default_model)
        self.settings = dict(DEFAULT_SETTINGS, brain_model=self.default_model)
        saved = self._load("settings.json", {})
        self.model_configured = False
        for key in FLAG_KEYS:
            if key in saved:
                self.settings[key] = bool(saved[key])
        for key, validate in VALIDATED_KEYS:
            if key in saved:
                try:
                    self.settings[key] = validate(saved[key])
                    if key == "brain_model":
                        self.model_configured = True
                except ValueError:
                    pass  # malformed hand-edited settings fall back independently
        for key in COMPACT_RANGES:
            if key in saved:
                try:
                    self.settings[key] = _compact_number(key, saved[key])
                except ValueError:
                    pass
        self._compacting = False              # a compaction call is running (in memory only)
        self.pose = "stand"                   # stand | sit | lie, from the world's NPC_POSE log (in memory only)
        self._affection_at = {}               # (player, type) -> time of the last note (in memory only)
        self.players = {}                     # name -> {"note", "locked", "updated", "talks"}
        for name, p in (self._load("impressions.json", {}).get("players") or {}).items():
            if isinstance(p, dict):
                self.players[_one_line(name, MAX_NAME_CHARS)] = self._clean_player(p)
        self.notes = [n for n in (self._clean_note(n) for n in self._load("memory.json", {}).get("notes") or []) if n]
        self.history = []                     # [{"role": "user"/"assistant", "content": str}, ...]
        if self.settings["keep_history"]:
            self.history = self._clean_history(self._load("history.json", []))

    def _migrate_data(self):
        """Publish a complete legacy copy atomically; retain originals and backups."""
        legacy_compact = os.path.join(self.root_dir, COMPACT_FILE)
        if os.path.exists(self.data_dir):
            # migrated by an older version that left the compaction undo backups behind: fill in only
            # when this NPC has no compaction file of its own yet
            target = os.path.join(self.data_dir, COMPACT_FILE)
            if os.path.isfile(legacy_compact) and not os.path.exists(target):
                try:
                    if not os.path.exists(legacy_compact + ".bak"):
                        shutil.copy2(legacy_compact, legacy_compact + ".bak")
                    shutil.copy2(legacy_compact, target + ".tmp")
                    os.replace(target + ".tmp", target)
                except OSError:
                    pass
            return
        names = [n for n in ("settings.json", "impressions.json", "memory.json", "history.json")
                 if os.path.isfile(os.path.join(self.root_dir, n))]
        if not names:
            return
        if os.path.isfile(legacy_compact):
            names.append(COMPACT_FILE)
        parent = os.path.dirname(self.data_dir)
        os.makedirs(parent, exist_ok=True)
        stage = tempfile.mkdtemp(prefix=".kwonjun-", dir=parent)
        try:
            for name in names:
                source = os.path.join(self.root_dir, name)
                backup = source + ".bak"
                if not os.path.exists(backup):
                    shutil.copy2(source, backup)
                shutil.copy2(source, os.path.join(stage, name))
            os.rename(stage, self.data_dir)
        finally:
            if os.path.isdir(stage):
                shutil.rmtree(stage)

    def for_npc(self, npc=None):
        selected = npc_id(npc)
        with self._registry_lock:
            if selected not in self._stores:
                child = Store(self.root_dir, npc=selected, default_model=self.default_model)
                child._stores, child._registry_lock = self._stores, self._registry_lock
                child.speaker, child.speaker_known = self.speaker, self.speaker_known
                self._stores[selected] = child
            return self._stores[selected]

    # ------------------------------------------------------------ files

    def _path(self, name: str) -> str:
        return os.path.join(self.data_dir, name)

    def _load(self, name: str, default):
        """A missing file is simply the default. A file that is there but cannot be used (unreadable, not
        JSON, wrong shape) is copied aside first, so the next save can never destroy the only copy; when
        even that copy fails, saving this file is switched off for this run (see _save)."""
        path = self._path(name)
        try:
            with open(path, encoding="utf-8-sig") as f:    # -sig: Notepad's "UTF-8 with BOM" reads fine
                data = json.load(f)
            if isinstance(data, type(default)):
                return data
        except FileNotFoundError:
            return default
        except (OSError, ValueError):
            pass
        if any(name == seen for seen, _ in self.load_problems):
            return default                    # already copied aside (or blocked) in this run
        backup = f"{path}.broken-{time.strftime('%Y%m%d-%H%M%S')}.bak"
        try:
            shutil.copy2(path, backup)
        except OSError:
            backup = ""
            self._blocked.add(name)
        self.load_problems.append((name, os.path.basename(backup)))
        return default

    def _save(self, name: str, data) -> None:
        if name in self._blocked:             # unreadable and not backed up: keep the file, change memory only
            return
        os.makedirs(self.data_dir, exist_ok=True)
        tmp = self._path(name) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path(name))     # never leaves a half-written file

    def _save_players(self):
        self._save("impressions.json", {"players": self.players})

    def _save_notes(self):
        self._save("memory.json", {"notes": self.notes})

    def _save_history(self):
        if self.settings["keep_history"]:
            self._save("history.json", self.history)
        else:
            try:
                os.remove(self._path("history.json"))
            except OSError:
                pass

    # ------------------------------------------------------------ cleaning (everything is capped)

    @staticmethod
    def _clean_player(p: dict) -> dict:
        try:
            talks = max(0, int(p.get("talks", 0)))
        except (TypeError, ValueError):
            talks = 0
        note = _one_line(p.get("note"), MAX_IMPRESSION_CHARS)
        recorded = p.get("touch_recorded")
        touch = [kind for kind in TOUCH_AFFECTION if AFFECTION_PHRASES[kind] in note
                 or (isinstance(recorded, list) and kind in recorded)]
        player = {"note": note, "locked": bool(p.get("locked")),
                  "updated": _one_line(p.get("updated"), 20), "talks": talks}
        if touch:
            player["touch_recorded"] = touch
        return player

    @staticmethod
    def _clean_note(n):
        if isinstance(n, str):
            n = {"text": n}
        if not isinstance(n, dict):
            return None
        text = _one_line(n.get("text"), MAX_NOTE_CHARS)
        if not text:
            return None
        return {"text": text, "by": "ai" if n.get("by") == "ai" else "user", "at": _one_line(n.get("at"), 20) or _stamp()}

    @staticmethod
    def _clean_history(items) -> list:
        out = []
        keys = ("speaker", "source", "time", "raw")
        for item in items:
            if (not isinstance(item, dict) or item.get("role") not in ("user", "assistant")
                    or not isinstance(item.get("content"), str)):
                continue
            message = {"role": item["role"], "content": item["content"]}
            meta = item.get("utterance")
            if message["role"] == "user" and isinstance(meta, dict) and all(isinstance(meta.get(key), str) for key in keys):
                message["utterance"] = {key: meta[key] for key in keys}
            out.append(message)
        pairs = [out[i:i + 2] for i in range(0, len(out) - 1, 2)
                 if out[i]["role"] == "user" and out[i + 1]["role"] == "assistant"]
        return [m for pair in pairs[-MAX_HISTORY:] for m in pair]

    # ------------------------------------------------------------ persona

    def _read_persona(self) -> str:
        try:
            with open(self.persona_path, encoding="utf-8-sig") as f:   # a BOM is not part of the text
                return f.read()
        except OSError:
            return ""

    def _renamed_persona(self, text: str) -> str:
        m = _NAME_RE.search(text)
        legacy, default = LEGACY_NPC_NAMES[self.npc], DEFAULT_NPC_NAMES[self.npc]
        if m and _one_line(m.group(1), MAX_NAME_CHARS) == legacy:
            text = text[:m.start(1)] + default + text[m.end(1):]
        text = re.sub(rf"(?m)^# {legacy}(?= 성격 설정(?: \(예시\))?$)", f"# {default}", text)
        if self.npc == "kwonjun":
            text = re.sub(r"(?m)^- 함께 사는 이토아는(?= 밝고 호기심이 많으며 귤을 좋아하는 친구다\.$)",
                          "- 함께 사는 ITOAR는", text)
        elif self.npc == "itoa":
            text = re.sub(r"(?m)^- 함께 사는 권준은(?= 장난기가 있고 밤 풍경과 영화를 좋아하는 친구다\.$)",
                          "- 함께 사는 kwonjun은", text)
        return text

    def _migrate_persona_names(self):
        raw = self._read_persona()
        renamed = self._renamed_persona(raw)
        if renamed != raw:
            try:
                self._write_persona(renamed)
            except OSError:
                pass  # read-only persona files still get the new defaults when read

    def persona_text(self) -> str:
        return self._renamed_persona(self._read_persona())

    def _write_persona(self, text: str) -> None:
        folder = os.path.dirname(self.persona_path)
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".persona-", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            os.replace(tmp, self.persona_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def save_persona(self, text: str) -> None:
        text = str(text).replace("\r\n", "\n")[:MAX_PERSONA_CHARS]
        with self.lock:
            self._write_persona(text)

    def npc_name(self) -> str:
        m = _NAME_RE.search(self.persona_text())
        return (_one_line(m.group(1), MAX_NAME_CHARS) if m else "") or DEFAULT_NPC_NAMES[self.npc]

    # ------------------------------------------------------------ shared NPC small talk

    @staticmethod
    def _smalltalk_from_data(data: dict) -> dict:
        settings = dict(DEFAULT_SMALLTALK)
        saved = data.get("settings")
        if isinstance(saved, dict):
            for key in settings:
                if key in saved:
                    try:
                        settings.update(_smalltalk_settings({key: saved[key]}))
                    except ValueError:
                        pass  # a malformed field falls back to its own default
        return settings

    def smalltalk_settings(self) -> dict:
        owner = self.for_npc()
        with owner.lock:
            return self._smalltalk_from_data(owner._load(SMALLTALK_STATE, {}))

    def set_smalltalk_settings(self, patch: dict) -> None:
        patch = _smalltalk_settings(patch)
        owner = self.for_npc()
        with owner.lock:
            data = owner._load(SMALLTALK_STATE, {})
            data["settings"] = dict(self._smalltalk_from_data(data), **patch)
            owner._save(SMALLTALK_STATE, data)

    def reserve_smalltalk_ai(self, draw: float) -> bool:
        owner = self.for_npc()
        with owner.lock:
            data = owner._load(SMALLTALK_STATE, {})
            settings = self._smalltalk_from_data(data)
            if not settings["ai"] or draw >= settings["ratio"]:
                return False
            today = time.strftime("%Y-%m-%d")
            used = data.get("used", 0) if data.get("day") == today else 0
            if type(used) is not int or used < 0:
                used = 0
            if used >= settings["daily_cap"]:
                return False
            data.update(day=today, used=used + 1)
            owner._save(SMALLTALK_STATE, data)
            return True

    # ------------------------------------------------------------ speaker / impressions / notes

    def set_speaker(self, name: str) -> None:
        name = _one_line(name, MAX_NAME_CHARS)
        if name:
            with self._registry_lock:
                for st in list(self._stores.values()):
                    st.speaker, st.speaker_known = name, True

    def impression(self, name: str) -> dict:
        with self.lock:
            return dict(self.players.get(name) or {"note": "", "locked": False, "updated": "", "talks": 0})

    def player_label(self, name: str) -> str:
        return f"{name}(나)" if self.speaker_known and name == self.speaker else name

    def _append_exchange(self, user: dict, assistant_json: str) -> None:
        self.history += [user, {"role": "assistant", "content": assistant_json}]
        self.history = self.history[-2 * MAX_HISTORY:]
        self._save_history()

    def record_overheard(self) -> None:
        with self.lock:
            self._append_exchange({"role": "user", "content": "(다른 사람에게 한 말로 판단)"},
                                  '{"for_me":false}')

    def after_reply(self, name: str, user_text: str, assistant_json: str, impression: str = "", memory_note: str = "", utterance=None) -> dict:
        """Called by the AI after a good reply: store the exchange and merge what it wants to remember."""
        changed = {}
        with self.lock:
            user = {"role": "user", "content": user_text}
            if utterance is not None:
                user["utterance"] = dict(utterance)
            self._append_exchange(user, assistant_json)
            p = self.players.get(name) or {"note": "", "locked": False, "updated": "", "talks": 0}
            p["talks"] += 1
            impression = _one_line(impression, AI_IMPRESSION_CHARS)
            if self.settings["auto_update"] and impression and not p["locked"] and impression != p["note"]:
                p["note"], p["updated"] = impression, _stamp()
                changed["impression"] = impression
            self.players[name] = p
            while len(self.players) > MAX_PLAYERS:       # forget whoever has the oldest unlocked note
                free = [k for k in self.players if k != name and not self.players[k]["locked"]]
                if not free:                             # everyone else is locked: the newcomer is not kept
                    del self.players[name]
                    changed.pop("impression", None)
                    break
                del self.players[min(free, key=lambda k: self.players[k]["updated"])]
            self._save_players()
            note = _one_line(memory_note, MAX_NOTE_CHARS)
            if self.settings["auto_update"] and note and all(n["text"] != note for n in self.notes):
                # full: the oldest note the AI wrote itself makes room; notes the user wrote are never dropped
                ai_notes = [n for n in self.notes if n["by"] == "ai"]
                while len(self.notes) >= MAX_NOTES and ai_notes:
                    self.notes.remove(ai_notes.pop(0))
                if len(self.notes) < MAX_NOTES:
                    self.notes.append({"text": note, "by": "ai", "at": _stamp()})
                    self._save_notes()
                    changed["memory_note"] = note
        return changed

    def add_affection(self, name: str, kind: str, now: float = None) -> str:
        """The world logged that `name` patted / hugged / waved at this NPC: add a short phrase to this
        NPC's impression of that player. Returns the phrase when it was written, else ''.
        Like the AI's own notes it needs "auto_update" on and the player's impression unlocked.
        Touch is recorded once per type even if the impression is later summarized."""
        phrase = AFFECTION_PHRASES.get(kind)
        name = _one_line(name, MAX_NAME_CHARS)
        if not phrase or not name:
            return ""
        now = time.time() if now is None else now
        with self.lock:
            if not self.settings["auto_update"]:
                return ""
            p = self.players.get(name)
            if p is None:
                if len(self.players) >= MAX_PLAYERS:      # never push out someone's impression for this
                    return ""
                p = {"note": "", "locked": False, "updated": "", "talks": 0}
            if p["locked"]:
                return ""
            recorded = p.get("touch_recorded", [])
            if kind in recorded:
                return ""
            last = self._affection_at.get((name, kind))
            if last is not None and 0 <= now - last < AFFECTION_INTERVAL_S:
                return ""
            self._affection_at[(name, kind)] = now
            note = (p["note"] + " " + phrase).strip()
            if phrase in p["note"] or len(note) > MAX_IMPRESSION_CHARS:
                return ""
            p["note"], p["updated"] = note, _stamp()
            if kind in TOUCH_AFFECTION:
                p["touch_recorded"] = [*recorded, kind]
            self.players[name] = p
            self._save_players()
            return phrase

    # ------------------------------------------------------------ what the AI is told

    def pick_notes(self, heard: str, speaker: str = "") -> list:
        """The memos worth showing for this utterance, in their stored order: those that share words with
        what was said (and with the previous thing the player said, for follow-up questions), the newest
        ones, and the newest hand-written ones. Local scoring only, no model call."""
        with self.lock:
            notes = self.notes
            if len(notes) <= PICK_ALL_UNDER:
                return list(notes)
            said = _bigrams(heard)
            if len(self.history) >= 2:
                said |= _bigrams(self.history[-2]["content"])
            name = _one_line(speaker, MAX_NAME_CHARS).lower()
            score = {}
            for i, n in enumerate(notes):
                hit = sum(len(piece) for piece in said & _bigrams(n["text"]))   # a two-letter match counts double
                if hit and name and name in n["text"].lower():
                    hit += 2
                if hit:
                    score[i] = hit
            best = sorted(score, key=lambda i: (-score[i], -i))[:PICK_RELEVANT]
            users = [i for i, n in enumerate(notes) if n["by"] == "user"][-PICK_USER:]
            chosen = set(best) | set(range(len(notes) - PICK_RECENT, len(notes))) | set(users)
            return [n for i, n in enumerate(notes) if i in chosen]

    def prompt_tail(self, speaker: str, heard: str = None) -> str:
        """Memory + impression block appended to the system prompt (small: capped notes, one player).
        heard given = only the memos picked for that utterance (pick_notes); None = all of them."""
        with self.lock:
            p = self.impression(speaker)
            lines = []
            if self.notes:
                shown = self.notes if heard is None else self.pick_notes(heard, speaker)
                lines.append("## 오래 기억하고 있는 것")
                lines += [f"- {n['text']}" for n in shown]
                if len(shown) < len(self.notes):
                    lines.append(f"- (지금 떠오른 것만 적었다. 메모가 {len(self.notes) - len(shown)}개 더 있다. "
                                 "그 안에 답이 있을 것 같으면 recall로 전부 되짚어 본다.)")
                lines.append("")
            lines.append("## 지금 말을 건 사람")
            label = self.player_label(speaker)
            lines.append(f"- 이름: {label}" if self.speaker_known or speaker != DEFAULT_SPEAKER else "- 이름은 아직 모른다.")
            if p["talks"]:
                lines.append(f"- 지금까지 {p['talks']}번 말을 주고받았다.")
            lines.append(f"- 이 사람에 대한 내 인상: {p['note']}" if p["note"]
                         else "- 이 사람에 대한 인상은 아직 없다.")
            return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ settings page

    def state(self) -> dict:
        smalltalk = self.smalltalk_settings()
        with self.lock:
            talks = []
            for i in range(0, len(self.history) - 1, 2):
                said = self.history[i + 1]["content"]
                try:
                    said = json.loads(said)["lines"]["ko"]
                except (ValueError, KeyError, TypeError):
                    pass
                user = self.history[i]
                talk = {"heard": user["content"], "line": said}
                speaker = user.get("utterance", {}).get("speaker")
                if speaker:
                    talk["speaker_label"] = self.player_label(speaker)
                talks.append(talk)
            return {
                "npc_name": self.npc_name(),
                "persona": self.persona_text(),
                "persona_file": os.path.basename(self.persona_path),
                "settings": dict(self.settings),
                "smalltalk": smalltalk,
                "model_options": model_options(),
                "effort_options": effort_options(),
                "speaker": self.speaker,
                "speaker_known": self.speaker_known,
                "players": {k: dict(v) for k, v in self.players.items()},
                "notes": [dict(n) for n in self.notes],
                "compaction": self.compaction_state(),
                "history": talks,
                "limits": {"persona": MAX_PERSONA_CHARS, "impression": MAX_IMPRESSION_CHARS, "note": MAX_NOTE_CHARS,
                           "notes": MAX_NOTES, "history": MAX_HISTORY, "name": MAX_NAME_CHARS},
            }

    def apply(self, req: dict) -> None:
        """One save from the settings page. Only the keys that are present are changed."""
        if not isinstance(req, dict):
            raise ValueError("bad request")
        smalltalk_patch = _smalltalk_settings(req["smalltalk"]) if "smalltalk" in req else None
        with self.lock:
            settings = None
            if isinstance(req.get("settings"), dict):
                settings = dict(self.settings)
                for key in FLAG_KEYS:
                    if key in req["settings"]:
                        settings[key] = bool(req["settings"][key])
                for key, validate in VALIDATED_KEYS:
                    if key in req["settings"]:
                        settings[key] = validate(req["settings"][key])
                for key in COMPACT_RANGES:
                    if key in req["settings"]:
                        settings[key] = _compact_number(key, req["settings"][key])
                # Validate the whole settings patch before changing any files or in-memory values.
            if isinstance(req.get("persona"), str):
                if not req["persona"].strip():
                    raise ValueError("성격 글이 비어 있어서 저장하지 않았습니다.")
                self.save_persona(req["persona"])
            if settings is not None:
                configured = self.model_configured or "brain_model" in req["settings"]
                persisted = dict(settings)
                if not configured:
                    del persisted["brain_model"]  # don't pin a CLI fallback by saving unrelated settings
                self._save("settings.json", persisted)
                self.settings = settings
                self.model_configured = configured
                if "keep_history" in req["settings"]:
                    self._save_history()      # writes or removes history.json to match the switch
            # Validate both lists before either is stored. "players_base" / "notes_base" are what the page
            # showed when the editing started: whatever the NPC added or changed since then is kept.
            merged_notes = None
            if isinstance(req.get("notes"), list):
                merged_notes = [n for n in (self._clean_note(n) for n in req["notes"][:MAX_NOTES]) if n]
                if isinstance(req.get("notes_base"), list):
                    seen = {_one_line(t, MAX_NOTE_CHARS) for t in req["notes_base"] if isinstance(t, str)}
                    seen |= {n["text"] for n in merged_notes}
                    merged_notes += [dict(n) for n in self.notes if n["text"] not in seen]
                    if len(merged_notes) > MAX_NOTES:
                        raise ValueError(f"고치는 사이에 NPC가 메모를 더 적어서 {MAX_NOTES}개를 넘습니다. "
                                         f"{len(merged_notes) - MAX_NOTES}개를 지운 뒤 다시 저장해 주세요.")
            if isinstance(req.get("players"), dict):
                new = {}
                for name, p in list(req["players"].items())[:MAX_PLAYERS]:
                    name = _one_line(name, MAX_NAME_CHARS)
                    if not name or not isinstance(p, dict):
                        continue
                    old = self.players.get(name, {})
                    clean = self._clean_player(dict(p, talks=old.get("talks", 0), updated=old.get("updated", ""),
                                                    touch_recorded=old.get("touch_recorded", [])))
                    if clean["note"] != old.get("note", ""):
                        clean["updated"] = _stamp()
                    new[name] = clean
                base = req.get("players_base")
                if isinstance(base, dict):
                    for name, cur in self.players.items():
                        seen = base.get(name)
                        if name not in new:
                            if seen is None:              # the NPC met this player while the page was open
                                new[name] = dict(cur)
                            continue                      # else: removed on the page
                        if not isinstance(seen, dict):
                            continue
                        mine = new[name]                  # a field the user left alone keeps the newer value
                        if (mine["note"] == _one_line(seen.get("note"), MAX_IMPRESSION_CHARS)
                                and cur["note"] != mine["note"]):
                            mine["note"], mine["updated"] = cur["note"], cur["updated"]
                        if mine["locked"] == bool(seen.get("locked")):
                            mine["locked"] = cur["locked"]
                    if len(new) > MAX_PLAYERS:
                        raise ValueError(f"고치는 사이에 사람이 늘어서 {MAX_PLAYERS}명을 넘습니다. "
                                         "몇 명을 지운 뒤 다시 저장해 주세요.")
                self.players = new
                self._save_players()
            if merged_notes is not None:
                self.notes = merged_notes
                self._save_notes()
            if req.get("history_clear"):
                self.history = []
                self._save_history()
            elif isinstance(req.get("history_delete"), int) and 0 <= req["history_delete"] < len(self.history) // 2:
                i = req["history_delete"] * 2
                if "history_heard" in req and self.history[i]["content"] != req["history_heard"]:
                    raise ValueError("그 사이에 대화가 바뀌었습니다. '새로 고침'을 누른 뒤 다시 해 주세요.")
                del self.history[i:i + 2]
                self._save_history()
        if smalltalk_patch is not None:
            self.set_smalltalk_settings(smalltalk_patch)

    # ------------------------------------------------------------ memory compaction
    # Like context compaction: when the memos grow past a threshold, ONE cheap model call merges and
    # condenses them. The caller (persona_npc_app.start_compaction) runs it on its own thread AFTER a
    # reply went out:   job = compaction_begin()  ->  model call, no lock held  ->  compaction_finish().

    def _compact_data(self) -> dict:
        data = self._load(COMPACT_FILE, {})
        if not isinstance(data.get("backups"), list):
            data["backups"] = []
        return data

    @staticmethod
    def _notes_sig(notes) -> list:
        return [n["text"] for n in notes]

    def _compactable(self, n: dict) -> bool:
        return n["by"] == "ai" or self.settings["compact_user_notes"]

    def _long_impressions(self) -> dict:
        limit = self.settings["compact_impression_chars"]
        found = {k: v["note"] for k, v in self.players.items() if not v["locked"] and len(v["note"]) > limit}
        longest = sorted(found, key=lambda k: -len(found[k]))[:COMPACT_MAX_IMPRESSIONS]
        return {k: found[k] for k in longest}

    def compaction_begin(self, force: bool = False, now: float = None):
        """-> (job, "") when a compaction should run now, else (None, Korean reason).
        force (the page's "지금 정리" button) skips the on/off switch, the thresholds and the interval."""
        now = time.time() if now is None else now
        with self.lock:
            s = self.settings
            if self._compacting:
                return None, "이미 정리하는 중입니다."
            if not force and not s["compact_on"]:
                return None, "자동 정리가 꺼져 있습니다."
            data = self._compact_data()
            last_try = data.get("last_try")
            if (not force and isinstance(last_try, (int, float))
                    and 0 <= now - last_try < s["compact_minutes"] * 60):
                return None, "방금 정리해서 아직 기다리는 중입니다."
            movable = [n for n in self.notes if self._compactable(n)]
            over = (len(self.notes) > s["compact_note_count"] or _chars(self.notes) > s["compact_note_chars"])
            # the same memos that the last run already produced (or could not shrink) are not sent again
            fresh = data.get("last_sig") != self._notes_sig(self.notes)
            notes_due = len(movable) >= 2 and (force or (over and fresh))
            people = self._long_impressions()
            if not notes_due and not people:
                return None, "줄일 만큼 쌓인 메모가 없습니다."
            job = {
                "npc": self.npc_name(),
                "notes": [dict(n) for n in movable] if notes_due else [],
                "keep": [n["text"] for n in self.notes if not self._compactable(n)] if notes_due else [],
                "impressions": people,
                "before": {"count": len(self.notes), "chars": _chars(self.notes)},
            }
            data["last_try"] = now
            self._save(COMPACT_FILE, data)      # a failed save must not leave "running" set for good
            self._compacting = True
            return job, ""

    def compaction_finish(self, job: dict, result=None, usage=None, error: str = "") -> dict:
        """Check and store what the model returned. Returns the summary that is also shown on the page.
        Memos and impressions edited while the call ran are never overwritten."""
        with self.lock:
            self._compacting = False
            data = self._compact_data()
            before = job["before"]
            last = {"at": _stamp(), "ok": False, "error": error, "notes_before": before, "notes_after": before,
                    "impressions": [], "usage": None, "cost_usd": 0.0}
            if usage:
                inp, out = MODEL_RATES[DEFAULT_MODEL][:2]
                last["usage"] = {"in": int(usage.get("in", 0)), "out": int(usage.get("out", 0))}
                last["cost_usd"] = round((last["usage"]["in"] * inp + last["usage"]["out"] * out) / 1e6, 6)
            if error or not isinstance(result, dict):
                last["error"] = error or "bad result"
                data["last"] = last
                self._save(COMPACT_FILE, data)
                return last

            old_notes = [dict(n) for n in self.notes]
            sent = {n["text"] for n in job["notes"]}
            gone = [n for n in self.notes if n["text"] in sent and self._compactable(n)]
            stay = [n for n in self.notes if not (n["text"] in sent and self._compactable(n))]
            taken = {n["text"] for n in stay}
            merged = []
            for text in (result.get("notes") or [])[:MAX_NOTES]:
                text = _one_line(text, MAX_NOTE_CHARS)
                if text and text not in taken:
                    taken.add(text)
                    merged.append({"text": text, "by": "ai", "at": _stamp()})
            # accept only a real reduction; an empty answer never wipes the memos
            notes_changed = bool(gone and merged and _chars(merged) < _chars(gone) and len(merged) <= len(gone))
            if notes_changed:
                known = sent | set(job["keep"])
                self.notes = ([n for n in stay if n["text"] in known] + merged
                              + [n for n in stay if n["text"] not in known])[:MAX_NOTES]

            people = {}
            for item in result.get("impressions") or []:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    continue
                name = item["name"]
                p, old = self.players.get(name), job["impressions"].get(name)
                new = _one_line(item.get("note"), MAX_IMPRESSION_CHARS)
                if old is None or p is None or p["locked"] or p["note"] != old or not new or len(new) >= len(old):
                    continue
                p["note"], p["updated"] = new, _stamp()
                people[name] = {"before": old, "after": new}

            if notes_changed or people:
                data["backups"] = (data["backups"] + [{
                    "at": last["at"], "notes": old_notes if notes_changed else None,
                    "after": self._notes_sig(self.notes) if notes_changed else None, "players": people,
                }])[-COMPACT_BACKUPS:]
                if notes_changed:
                    self._save_notes()
                if people:
                    self._save_players()
                last["ok"] = True
            else:
                last["error"] = "줄일 것이 없었음"
            last["notes_after"] = {"count": len(self.notes), "chars": _chars(self.notes)}
            last["impressions"] = [{"name": k, "before": len(v["before"]), "after": len(v["after"])}
                                   for k, v in people.items()]
            data["last"], data["last_sig"] = last, self._notes_sig(self.notes)
            self._save(COMPACT_FILE, data)
            return last

    def compaction_undo(self) -> str:
        """Put back what the newest backup holds ("되돌리기"). '' when done, else the Korean reason.
        Memos written after that compaction stay; an impression edited since then is left alone.
        Automatic compaction is switched off, or the next reply would condense the same memos again."""
        with self.lock:
            if self._compacting:
                return "지금 정리하는 중입니다. 잠시 뒤에 다시 눌러 주세요."
            data = self._compact_data()
            if not data["backups"]:
                return "되돌릴 정리 기록이 없습니다."
            b = data["backups"][-1]
            if not isinstance(b, dict):
                b = {}
            restored = None
            if isinstance(b.get("notes"), list):
                old = [n for n in (self._clean_note(n) for n in b["notes"]) if n]
                skip = set(b.get("after") or []) | {n["text"] for n in old}
                restored = old + [n for n in self.notes if n["text"] not in skip]
                if len(restored) > MAX_NOTES:   # nothing is changed: no memo written since then is thrown away
                    return (f"되돌리면 메모가 {len(restored)}개가 되어 최대 {MAX_NOTES}개를 넘습니다. "
                            f"지금 메모를 {len(restored) - MAX_NOTES}개 지운 뒤 다시 눌러 주세요.")
            data["backups"].pop()
            if restored is not None:
                self.notes = restored
                self._save_notes()
            changed = False
            for name, v in (b.get("players") or {}).items():
                p = self.players.get(name)
                if p and isinstance(v, dict) and p["note"] == v.get("after"):
                    p["note"], p["updated"], changed = _one_line(v.get("before"), MAX_IMPRESSION_CHARS), _stamp(), True
            if changed:
                self._save_players()
            data["last_sig"] = self._notes_sig(self.notes)
            if isinstance(data.get("last"), dict):
                data["last"]["undone"] = _stamp()
            self._save(COMPACT_FILE, data)
            self.settings = dict(self.settings, compact_on=False)
            persisted = dict(self.settings)
            if not self.model_configured:
                del persisted["brain_model"]
            self._save("settings.json", persisted)
        return ""

    def compaction_state(self) -> dict:
        with self.lock:
            data = self._compact_data()
            last = data.get("last") if isinstance(data.get("last"), dict) else None
            return {"running": self._compacting, "last": last,
                    "backups": [b.get("at", "") for b in data["backups"] if isinstance(b, dict)],
                    "now": {"count": len(self.notes), "chars": _chars(self.notes)},
                    "ranges": {k: list(v) for k, v in COMPACT_RANGES.items()}}

    # ------------------------------------------------------------ single edits (in-world panel, cfg_bridge.py)
    # Each returns "" when applied, or a short Korean reason when refused. Same caps as apply() above.

    def set_persona(self, text) -> str:
        if not isinstance(text, str) or not text.strip():
            return "성격 글이 비어 있어서 저장하지 않았습니다."
        self.save_persona(text)
        return ""

    def append_persona(self, text) -> str:
        text = _one_line(text, MAX_PERSONA_CHARS)
        if not text:
            return "덧붙일 말이 비어 있습니다."
        with self.lock:
            new = self.persona_text().replace("\r\n", "\n").rstrip() + "\n- " + text + "\n"
            if len(new) > MAX_PERSONA_CHARS:
                return f"성격 글이 {MAX_PERSONA_CHARS}자를 넘어서 덧붙이지 않았습니다."
            self.save_persona(new)
        return ""

    def set_npc_name(self, name) -> str:
        name = _one_line(name, MAX_NAME_CHARS)
        if not name:
            return "이름이 비어 있습니다."
        with self.lock:
            text = self.persona_text().replace("\r\n", "\n")
            m = _NAME_RE.search(text)
            if m:
                text = text[:m.start(1)] + name + text[m.end(1):]
            else:
                text = text.rstrip() + "\n- 이름: " + name + "\n"
            if len(text) > MAX_PERSONA_CHARS:
                return f"성격 글이 {MAX_PERSONA_CHARS}자를 넘어서 이름을 바꾸지 않았습니다."
            self.save_persona(text)
        return ""

    def _player_for_edit(self, name):
        """(clean name, entry) for a manual edit, or (None, reason). Unknown names get a new entry."""
        name = _one_line(name, MAX_NAME_CHARS) or self.speaker
        p = self.players.get(name)
        if p is None:
            if len(self.players) >= MAX_PLAYERS:
                return None, f"플레이어가 {MAX_PLAYERS}명이라 더 적을 수 없습니다."
            p = {"note": "", "locked": False, "updated": "", "talks": 0}
        return name, p

    def set_impression(self, name, text) -> str:
        with self.lock:
            name, p = self._player_for_edit(name)
            if name is None:
                return p
            note = _one_line(text, MAX_IMPRESSION_CHARS)
            if note != p["note"]:
                p["note"], p["updated"] = note, _stamp()
            self.players[name] = p
            self._save_players()
        return ""

    def set_locked(self, name, on) -> str:
        with self.lock:
            name, p = self._player_for_edit(name)
            if name is None:
                return p
            p["locked"] = bool(on)
            self.players[name] = p
            self._save_players()
        return ""

    def add_note(self, text) -> str:
        note = self._clean_note({"text": text, "by": "user"})
        if not note:
            return "기억 메모가 비어 있습니다."
        with self.lock:
            if len(self.notes) >= MAX_NOTES:
                return f"기억 메모가 {MAX_NOTES}개라 더 넣을 수 없습니다. 먼저 하나를 지워 주세요."
            self.notes.append(note)
            self._save_notes()
        return ""

    def edit_note(self, index, text, expect=None) -> str:
        """text '' = delete (the web page drops empty notes too). expect = the note text the editor saw."""
        with self.lock:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.notes):
                return "그 번호의 기억 메모가 없습니다. 새로 고친 뒤 다시 해 주세요."
            if isinstance(expect, str) and _one_line(expect, MAX_NOTE_CHARS) != self.notes[index]["text"]:
                return "그 사이에 기억 메모가 바뀌었습니다. 새로 고친 뒤 다시 해 주세요."
            text = _one_line(text, MAX_NOTE_CHARS)
            if text:
                if text != self.notes[index]["text"]:
                    self.notes[index] = {"text": text, "by": "user", "at": _stamp()}
            else:
                del self.notes[index]
            self._save_notes()
        return ""
