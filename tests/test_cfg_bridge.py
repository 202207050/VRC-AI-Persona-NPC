"""Offline check of the in-world settings panel bridge (cfg_bridge.py) and the log watcher.

    .venv\\Scripts\\python.exe tests\\test_cfg_bridge.py

No microphone, no Whisper, no API call, no real VRChat log: fake logs live in a temporary folder, the
server runs on a spare port, speech-to-text and the Claude client are fakes.
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)

import cfg_bridge              # noqa: E402
import persona_ai              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402

checks = 0
LOCAL = "local:1a7a51216f56f32584745f9ce2887998a0e3a76e83f7e20749da576bb20137da"
MINE = "wrld_11111111-2222-3333-4444-555555555555"
OTHER = "wrld_b05ed97c-0856-4c72-bda6-c1a99bae30be"
T = "2026.11.19 13:44:37 Debug      -  "


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def join(world):
    tail = "" if world.startswith("local:") else ":59823~private(usr_86bc47ca-4221-47cf-b5cd-b2341fd4c964)~region(jp)"
    return (T + "[Behaviour] Entering Room: Some Room\n" + T + f"[Behaviour] Joining {world}{tail}\n"
            + T + "[Behaviour] Joining or Creating Room: Some Room\n")


def cfg_set(**req):
    """As Udon writes it: VRCJson on one line, then the probe's ' utc=' tail."""
    return T + "[W04AI] CFG_SET " + json.dumps(req, ensure_ascii=bool(req.pop("_ascii", False))) + " utc=639306602829378126\n"


CFG_OPEN = T + "[W04AI] CFG_OPEN utc=639306602829378126\n"
PTT_DOWN = T + "[W04AI] PTT_DOWN utc=1\n"
PTT_UP = T + "[W04AI] PTT_UP held_ms=1500 utc=2\n"
PTT = PTT_DOWN + PTT_UP


def get(port, path, headers=None, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path, headers=headers or {})
    res = conn.getresponse()
    raw = res.read()
    conn.close()
    return res.status, raw


def wait_for(cond, what, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out: " + what)


class FakeRec:
    name = "fake mic"

    def start(self):
        pass

    def stop(self):
        return (np.random.default_rng(0).standard_normal(app.RATE) * 0.1).astype(np.float32)


class FakeTr:
    """Stands in for Whisper: every clip 'says' self.text."""
    text = ""

    def run(self, audio):
        return {"text": self.text, "segments": []}


class CountingBrain:
    """Stands in for the AI: must NOT be called on a dictation turn."""

    def __init__(self, store):
        self.store, self.model, self.heard = store, "mock", []

    def reply(self, text, **kw):
        self.heard.append(text)
        return {"emotion": "happy", "intensity": 0.5, "steps": [], "lines": {"ko": "응!"}, "line": "응!",
                "ai_ms": 1, "model": "mock"}


def world_allowlist_checks(tmp):
    example_path = os.path.join(tmp, ".env.example")
    ok(cfg_bridge.load_world_ids(MINE, example_path) == {MINE}, "missing example keeps configured worlds")
    with open(example_path, "w", encoding="utf-8-sig") as example:
        example.write(f'# W04_WORLD_IDS=wrld_bad\nOTHER_KEY=wrld_bad\n W04_WORLD_IDS = "{MINE}, {OTHER}, invalid"\n')

    personal = "wrld_aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    ok(cfg_bridge.load_world_ids(f"{MINE},{personal}", example_path) == {MINE, OTHER, personal},
       "the allowlist combines both sources, filters invalid IDs and counts overlaps once")
    ok(cfg_bridge.load_world_ids("", example_path) == {MINE, OTHER},
       "the bundled worlds work without a private setting")


def unit_checks(tmp):
    """The bridge alone, with a hand-moved clock."""
    persona = os.path.join(tmp, "persona_unit.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    store = persona_store.Store(data_dir=os.path.join(tmp, "data_unit"), persona_path=persona)
    store.set_speaker("Tester One")
    now = [1000.0]
    logged = []
    bridge = cfg_bridge.CfgBridge(store, cfg_bridge.parse_world_ids(f" {MINE} , nonsense,"),
                                  log=lambda kind, detail="", say="": logged.append((kind, detail, say)),
                                  clock=lambda: now[0])

    # ---- which worlds count
    ok(bridge.world_allowed(LOCAL) and bridge.world_allowed(MINE), "Build & Test and listed IDs are allowed")
    ok(not bridge.world_allowed(OTHER) and not bridge.world_allowed(None) and not bridge.world_allowed(""),
       "other worlds / no world are not")
    world = None
    for line in join(OTHER).splitlines():
        world = cfg_bridge.world_from_line(line, world)
    ok(world == OTHER, "world ID read from VRChat's 'Joining wrld_...:instance' line")
    for line in join(LOCAL).splitlines():
        world = cfg_bridge.world_from_line(line, world)
    ok(world == LOCAL, "local test world read from 'Joining local:...'")
    ok(cfg_bridge.world_from_line(T + "[Behaviour] OnLeftRoom", world) is None, "OnLeftRoom -> no world")
    ok(cfg_bridge.world_from_line("x\n[Behaviour] Joining " + MINE, OTHER) == OTHER
       and cfg_bridge.world_from_line(T + "[Video] url [Behaviour] Joining " + MINE, OTHER) == OTHER,
       "a join text in the middle of another line does not count")

    # ---- window + world gating of CFG lines
    ok(not bridge.is_open(), "closed at start")
    ok(bridge.handle_line("a", OTHER, CFG_OPEN) and not bridge.is_open(), "CFG_OPEN from a foreign world is ignored")
    ok(bridge.handle_line("a", None, CFG_OPEN) and not bridge.is_open(), "CFG_OPEN with unknown world is ignored")
    before = store.persona_text()
    bridge.handle_line("a", OTHER, cfg_set(op="persona", text="해킹"))
    ok(store.persona_text() == before, "CFG_SET from a foreign world is ignored")
    ok(not bridge.handle_line("a", LOCAL, T + "[W04AI] PTT_DOWN utc=1"), "other [W04AI] lines are left to the app")
    ok(not bridge.handle_line("a", LOCAL, T + "[Video] title: [W04AI] CFG_OPEN") and not bridge.is_open(),
       "CFG text inside another line does not count")
    bridge.handle_line("a", LOCAL, CFG_OPEN)
    ok(bridge.is_open(), "CFG_OPEN from an allowed world opens the window")
    now[0] += 19
    ok(bridge.is_open(), "still open after 19 s")
    bridge.handle_line("a", LOCAL, CFG_OPEN)
    now[0] += 19
    ok(bridge.is_open(), "a second CFG_OPEN restarts the 20 s")
    now[0] += 2
    ok(not bridge.is_open(), "closed 20 s after the last CFG_OPEN")
    bridge.handle_line("a", MINE, CFG_OPEN)
    bridge.world_changed("b")
    ok(bridge.is_open(), "another client changing world does not close this window")
    bridge.world_changed("a")
    ok(not bridge.is_open(), "leaving the world closes the window at once")

    # ---- every op
    def send(**req):
        bridge.handle_line("a", LOCAL, cfg_set(**req))
        return bridge.msg

    def cfg():
        return json.loads(bridge.payload().decode("utf-8"))

    rev0 = cfg()["rev"]
    ok(cfg()["rev"] == rev0, "rev stays while nothing changes")
    ok(send(op="persona", text="# 루나\n- 이름: 루나\n- 조용한 아이", _ascii=True) == "", "persona (\\u escapes, \\n)")
    ok(store.persona_text() == "# 루나\n- 이름: 루나\n- 조용한 아이" and cfg()["npc"] == "루나", "persona stored")
    ok(cfg()["rev"] == rev0 + 1, "rev +1 after a change")
    ok(send(op="persona", text="  ") != "" and "조용한" in store.persona_text(), "empty persona refused")
    send(op="persona", text="가" * 9000)
    ok(len(store.persona_text()) == persona_store.MAX_PERSONA_CHARS, "persona capped like the web page")
    c = cfg()
    ok(len(bridge.payload()) <= cfg_bridge.MAX_CFG_BYTES and c.get("persona_cut") is True, "/cfg stays under 16 KB")
    ok(send(op="persona", text="일부만 보고 저장") != "" and len(store.persona_text()) == persona_store.MAX_PERSONA_CHARS,
       "a cut persona cannot be saved back from the panel")
    store.save_persona("# 루나\n- 이름: 루나\n- 조용한 아이\n")
    ok("persona_cut" not in cfg(), "short persona is sent whole")

    ok(send(op="name", text=" 미나 \n") == "" and store.npc_name() == "미나" and "조용한 아이" in store.persona_text(),
       "name changes only the name line")
    ok(send(op="name", text="") != "" and store.npc_name() == "미나", "empty name refused")

    ok(send(op="impression", player="Tester One", text="다" * (persona_store.MAX_IMPRESSION_CHARS + 100)) == "", "impression")
    ok(len(store.players["Tester One"]["note"]) == persona_store.MAX_IMPRESSION_CHARS, "hand-written impression capped")
    send(op="impression", text="내 인상")
    ok(store.players["Tester One"]["note"] == "내 인상", "no player -> the current speaker")
    send(op="lock", player="Tester One", on=True)
    ok(store.players["Tester One"]["locked"] is True, "lock on")
    send(op="lock", player="Tester One", on=False)
    ok(store.players["Tester One"]["locked"] is False, "lock off")
    send(op="lock", player="Friend", on=True)
    ok(store.players["Friend"] == dict(store.players["Friend"], note="", locked=True), "lock makes a new entry")

    ok(send(op="note_add", text="라" * 300) == "" and len(store.notes[0]["text"]) == persona_store.MAX_NOTE_CHARS,
       "note capped at 80")
    ok(store.notes[0]["by"] == "user", "panel notes are the user's")
    send(op="note_edit", index=0.0, text="손님은 커피를 좋아한다")          # VRCJson numbers are doubles
    ok(store.notes[0]["text"] == "손님은 커피를 좋아한다", "note_edit")
    for i in range(persona_store.MAX_NOTES + 10):
        send(op="note_add", text=f"메모 {i}")
    ok(len(store.notes) == persona_store.MAX_NOTES and bridge.msg != "", "notes capped, the rest refused")
    ok(send(op="note_edit", index=99, text="x") != "" and send(op="note_del", index=-1) != ""
       and send(op="note_del", index="0") != "", "bad index refused")
    ok(send(op="note_del", index=1, expect="다른 글") != "" and len(store.notes) == persona_store.MAX_NOTES,
       "stale delete refused when 'expect' does not match")
    ok(send(op="note_del", index=1, expect="메모 0") == "" and store.notes[1]["text"] == "메모 1", "note_del")
    send(op="note_edit", index=1, text=" ")
    ok(store.notes[1]["text"] == "메모 2", "note_edit with empty text removes the note (as the web page does)")

    # memo editor in the panel: the whole list as lines (2026-11-19)
    store.apply({"notes": [{"text": "고양이를 좋아함", "by": "ai"}, {"text": "커피는 싫어함", "by": "user"}]})
    start = "고양이를 좋아함\n커피는 싫어함"
    store.add_note("AI가 그 사이 적음")                 # written while the player was editing
    ok(send(op="notes_set", text="고양이를 좋아함\r\n\n  새로 쓴 메모  \n", expect=start) == "", "notes_set applied")
    ok([n["text"] for n in store.notes] == ["고양이를 좋아함", "새로 쓴 메모", "AI가 그 사이 적음"],
       "deleted line removed, new line added, memo written meanwhile kept")
    ok(store.notes[0]["by"] == "ai" and store.notes[1]["by"] == "user", "kept lines keep their author")
    ok(send(op="notes_set", text="", expect="고양이를 좋아함\n새로 쓴 메모\nAI가 그 사이 적음") == "" and store.notes == [],
       "empty editor clears all memos")
    ok(send(op="notes_set") != "", "notes_set without text refused")

    store.after_reply("Tester One", "안녕", json.dumps({"lines": {"ko": "어서 와!"}}, ensure_ascii=False))
    ok(cfg()["history"] == [{"who": "user", "text": "안녕"}, {"who": "npc", "text": "어서 와!"}], "history view")
    for i in range(8):
        store.after_reply("Tester One", f"말 {i}", json.dumps({"lines": {"ko": f"답 {i}"}}, ensure_ascii=False))
    h = cfg()["history"]
    ok(len(h) == 10 and h[-1] == {"who": "npc", "text": "답 7"}, "history: last 10 lines")
    send(op="keep_history", on=True)
    ok(store.settings["keep_history"] and os.path.exists(os.path.join(store.data_dir, "history.json")), "keep_history on")
    send(op="history_clear")
    ok(store.history == [] and cfg()["history"] == [], "history_clear")
    send(op="keep_history", on=False)
    send(op="auto", on=False)
    ok(not store.settings["auto_update"] and not store.settings["keep_history"] and cfg()["auto"] is False, "auto / keep_history off")
    send(op="auto", on=True)
    ok(cfg()["auto"] is True, "auto on")
    ok(send(op="format_disk") != "" and send(op="persona") != "", "unknown op / missing text refused")
    bridge.handle_line("a", LOCAL, T + "[W04AI] CFG_SET {broken utc=1")
    bridge.handle_line("a", LOCAL, T + "[W04AI] CFG_SET [1,2] utc=1")
    ok(logged[-1][1].startswith("bad_json"), "broken JSON is ignored")

    c = cfg()
    ok(set(c) == {"ok", "rev", "npc", "speaker", "persona", "auto", "keep_history", "smalltalk", "players", "notes", "history",
                  "dictate", "msg", "act", "errors", "memory"}, "/cfg keys")
    ok(set(c["memory"]) == {"v", "persona", "people", "notes", "changed_at", "changes", "digest"}
       and c["memory"]["people"][0]["name"] == "Tester One" and c["memory"]["people"][0]["talks"] == 9,
       "/cfg memory (t20): the people the NPC talked with, newest first")
    ok(isinstance(c["errors"], list), "/cfg errors (TODO 140) is a list")
    ok(c["players"][0] == {"name": "Tester One", "note": "내 인상", "locked": False} and c["speaker"] == "Tester One",
       "players: the speaker first")
    ok(all(isinstance(n, str) for n in c["notes"]) and c["dictate"] is None, "notes are plain texts")
    for i in range(persona_store.MAX_PLAYERS + 5):
        send(op="impression", player=f"P{i}", text="마" * 150)
    ok(len(store.players) == persona_store.MAX_PLAYERS and len(bridge.payload()) <= cfg_bridge.MAX_CFG_BYTES,
       "players capped, /cfg still under 16 KB")
    memory = cfg()["memory"]
    ok(len(memory["people"]) <= cfg_bridge.MEMORY_PEOPLE and all(len(p["note"]) <= cfg_bridge.CARD_CHARS for p in memory["people"])
       and len(memory["digest"].encode("utf-8")) <= cfg_bridge.DIGEST_MAX_BYTES
       and memory["digest"].startswith(cfg_bridge.DIGEST_HEAD + "\nT\t"), "memory block and digest stay within their limits")

    # ---- dictation state
    send(op="dictate", target="note_add")
    ok(cfg()["dictate"] == {"target": "note_add", "player": ""}, "dictate armed")
    now[0] += 61
    ok(bridge.dictate_target() is None and cfg()["dictate"] is None, "dictate expires after 60 s")
    ok(send(op="dictate", target="shell") != "" and bridge.dictate_target() is None, "unknown dictate target refused")
    send(op="dictate", target="impression")
    ok(bridge.dictate_target()["player"] == "Tester One", "impression dictation defaults to the speaker")
    bridge.world_changed("a")
    ok(bridge.dictate_target() is None, "leaving the world disarms dictation")
    send(op="dictate", target="persona_append")
    ok(bridge.dictate_apply("밤 산책을 좋아한다") == "" and store.persona_text().endswith("- 밤 산책을 좋아한다\n"),
       "persona_append adds a line")
    ok(bridge.dictate_target() is None, "used once")

    # ---- the log never gets the text
    text_seen = [d for k, d, s in logged if any(w in d + s for w in ("조용한", "커피", "산책", "내 인상", "미나"))]
    ok(not text_seen and any(d.startswith("op=persona\tlen=") and d.endswith("\tok") for k, d, s in logged),
       "edits are logged as op + length only")


def watcher_checks(tmp):
    """Two fake VRChat logs, the real watcher loop, the real HTTP server."""
    persona = os.path.join(tmp, "persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    store = persona_store.Store(data_dir=os.path.join(tmp, "data"), persona_path=persona)
    bridge = cfg_bridge.CfgBridge(store, {MINE}, log=app.log)
    brain, tr = CountingBrain(store), FakeTr()
    logs = os.path.join(tmp, "vrc")
    os.makedirs(logs)
    log_a = os.path.join(logs, "output_log_2026-11-19_13-44-33.txt")    # Build & Test client
    log_b = os.path.join(logs, "output_log_2026-11-19_13-43-58.txt")    # another client in someone's world
    old = os.path.join(logs, "output_log_2026-11-11_22-51-25.txt")      # last week's
    with open(log_a, "w", encoding="utf-8") as f:
        f.write(join(LOCAL) + T + '[Behaviour] Initialized PlayerAPI "Tester One" is local\n'
                + CFG_OPEN + cfg_set(op="note_add", text="예전 줄") + cfg_set(op="dictate", target="note_add") + PTT)
    with open(log_b, "w", encoding="utf-8") as f:
        f.write(join(OTHER) + CFG_OPEN)
    with open(old, "w", encoding="utf-8") as f:
        f.write(join(LOCAL) + CFG_OPEN)
    os.utime(old, (time.time() - 86400, time.time() - 86400))
    ok([os.path.basename(p) for p in app.recent_output_logs(logs)] == [os.path.basename(log_b), os.path.basename(log_a)],
       "both recent logs are followed, last week's is not")

    def add(path, text):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    def tsv(kind):
        with open(app.LOG_PATH, encoding="utf-8") as f:
            return [line.rstrip("\n").split("\t", 2)[2] for line in f if line.split("\t")[1] == kind]

    while not app.replies.empty():
        app.replies.get_nowait()
    stop = threading.Event()
    watcher = threading.Thread(target=app.watch_log, args=(logs, FakeRec(), tr, False, brain, store, bridge, stop), daemon=True)
    watcher.start()
    server = app.make_server(0, store, brain, bridge=bridge)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg_bridge.OPEN_GRACE_S = 0.0
    try:
        wait_for(lambda: store.speaker == "Tester One", "watcher read the backlog")
        time.sleep(0.3)
        # ---- nothing from before the start is replayed
        ok(store.notes == [] and bridge.dictate_target() is None and not bridge.is_open() and app.replies.empty(),
           "old CFG_OPEN / CFG_SET / PTT lines are not applied again")
        ok(get(port, "/cfg")[0] == 403 and get(port, "/cfg")[1] == b'{"ok":false}', "/cfg closed -> 403, tiny body")

        # ---- foreign world (second log): everything ignored
        add(log_b, CFG_OPEN + cfg_set(op="note_add", text="남의 월드") + PTT)
        wait_for(lambda: len(tsv("CFG_IGNORED")) == 2 and tsv("PTT_IGNORED"), "foreign lines seen")
        ok(store.notes == [] and get(port, "/cfg")[0] == 403 and app.replies.empty() and not brain.heard,
           "CFG and PTT from a log whose world is not allowed do nothing")

        # ---- allowed world (first log)
        add(log_a, CFG_OPEN)
        wait_for(bridge.is_open, "window opened")
        status, raw = get(port, "/cfg")
        c = json.loads(raw.decode("utf-8"))
        ok(status == 200 and c["ok"] is True and c["npc"] == "kwonjun" and c["speaker"] == "Tester One", "/cfg open, no token")
        ok(b"\\u" not in raw and b", " not in raw.split(b'"persona"')[0], "compact UTF-8 JSON")
        ok(get(port, "/cfg", {"Host": f"evil.example:{port}"})[0] == 403, "/cfg with a foreign Host -> 403")
        ok(get(port, "/cfg", {"Origin": "http://evil.example"})[0] == 403, "/cfg from a web page (Origin) -> 403")
        ok(get(port, "/api/state")[0] == 403, "the token-protected page data is still closed")

        cfg_bridge.OPEN_GRACE_S = 3.0                       # /cfg may arrive just before the CFG_OPEN line
        bridge.world_changed(log_a)
        threading.Timer(0.4, add, (log_a, CFG_OPEN)).start()
        ok(get(port, "/cfg")[0] == 200, "/cfg waits a moment for a late CFG_OPEN line")
        cfg_bridge.OPEN_GRACE_S = 0.0

        add(log_a, cfg_set(op="note_add", text="손님은 커피를 좋아한다"))
        wait_for(lambda: len(store.notes) == 1, "live CFG_SET applied")
        rev = c["rev"]
        ok(json.loads(get(port, "/cfg")[1])["rev"] == rev + 1, "rev grows by one per change")

        # ---- normal push-to-talk still goes to the AI
        tr.text = "안녕 반가워"
        add(log_a, PTT_DOWN)
        status, raw = get(port, "/poll")
        ok(status == 200 and json.loads(raw) == {"state": "hearing", "speaker": "Tester One"},
           "push-to-talk can show hearing while it is held")
        add(log_a, PTT_UP)
        status, raw = get(port, "/poll")
        ok(json.loads(raw)["line"] == "응!" and brain.heard == ["안녕 반가워"],
           "normal turn: the next poll gets the answer without an extra thinking status")

        # ---- dictation: the next push-to-talk goes to the target, not to the AI
        add(log_a, cfg_set(op="dictate", target="note_add"))
        wait_for(lambda: bridge.dictate_target() is not None, "dictate armed")
        tr.text = "루나는 딸기를 좋아한다"
        add(log_a, PTT)
        status, raw = get(port, "/poll")
        reply = json.loads(raw)
        ok(status == 200 and reply.get("line", "") == "" and reply["dictate"] == "note_add",
           "/poll answers the dictation turn with no line (the world's POLL_EMPTY)")
        ok(store.notes[-1]["text"] == "루나는 딸기를 좋아한다" and brain.heard == ["안녕 반가워"],
           "transcript went to the notes, the AI was not called")
        ok(bridge.dictate_target() is None, "dictation used up")
        ok(not any("딸기" in line for line in tsv("STT") + tsv("CFG_DICTATE") + tsv("CFG_SET")),
           "dictated text is not written to voice_log.tsv")

        add(log_a, cfg_set(op="dictate", target="impression"))
        wait_for(lambda: bridge.dictate_target() is not None, "dictate armed again")
        tr.text = ""                                         # nothing understood: stays armed, world still gets an answer
        add(log_a, PTT)
        ok(json.loads(get(port, "/poll")[1]).get("line", "") == "" and bridge.dictate_target() is not None,
           "a missed clip keeps the dictation waiting")
        tr.text = "다정하고 말이 많다"
        add(log_a, PTT)
        get(port, "/poll")
        ok(store.players["Tester One"]["note"] == "다정하고 말이 많다" and len(brain.heard) == 1, "impression dictated")
        add(log_a, PTT_DOWN)
        ok(json.loads(get(port, "/poll")[1])["state"] == "hearing", "the turn after dictation reports hearing")
        add(log_a, PTT_UP)
        ok(json.loads(get(port, "/poll")[1])["line"] == "응!" and len(brain.heard) == 2, "the turn after goes to the AI again")

        # ---- half-written line, then the rest
        add(log_a, T + "[W04AI] CFG_SET " + json.dumps({"op": "note_add", "text": "둘로 나뉜 줄"})[:20])
        time.sleep(0.2)
        add(log_a, json.dumps({"op": "note_add", "text": "둘로 나뉜 줄"})[20:] + " utc=1\n")
        wait_for(lambda: store.notes[-1]["text"] == "둘로 나뉜 줄", "a line written in two parts is read whole")

        # ---- worlds are tracked per file
        add(log_a, T + "[Behaviour] OnLeftRoom\n" + join(OTHER))
        add(log_b, T + "[Behaviour] OnLeftRoom\n" + join(MINE))
        wait_for(lambda: len(tsv("WORLD")) >= 6, "both clients changed world")
        n = len(store.notes)
        add(log_a, cfg_set(op="note_add", text="이제 남의 월드") + CFG_OPEN)
        add(log_b, cfg_set(op="note_add", text="이제 내 월드"))
        wait_for(lambda: len(store.notes) == n + 1, "second log is now the allowed one")
        time.sleep(0.2)
        ok(store.notes[-1]["text"] == "이제 내 월드" and len(store.notes) == n + 1 and not bridge.is_open(),
           "allowed world is decided per log file")

        ok(len(tsv("WATCH")) == 1, "WATCH is logged once, not on every switch between the two files")
        new = os.path.join(logs, "output_log_2026-11-19_14-00-00.txt")
        with open(new, "w", encoding="utf-8") as f:
            f.write(join(LOCAL) + cfg_set(op="note_add", text="새 파일의 예전 줄"))
        wait_for(lambda: len(tsv("WATCH")) == 2, "a third log joins the set")
        n = len(store.notes)
        add(new, cfg_set(op="note_add", text="새 파일의 새 줄"))
        wait_for(lambda: len(store.notes) == n + 1, "new file followed from its end")
        ok(store.notes[-1]["text"] == "새 파일의 새 줄", "a file seen for the first time starts at its end")
    finally:
        cfg_bridge.OPEN_GRACE_S = 2.0
        stop.set()
        server.shutdown()
        server.server_close()
        watcher.join(timeout=3)


def main():
    tmp = tempfile.mkdtemp(prefix="w04_cfg_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")        # keep the real log clean
        world_allowlist_checks(tmp)
        unit_checks(tmp)
        watcher_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_regression_checks():
    main()


if __name__ == "__main__":
    main()
