"""Offline checks for the fixes from the 2026-11-20 review triage.

    .venv\\Scripts\\python.exe -X utf8 tests\\test_triage_fixes.py

No microphone, no Whisper, no API call, no real VRChat log (same fakes as test_gaze_listen.py).
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("PERSONA_FILE", None)

import cfg_bridge              # noqa: E402
import persona_ai              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402
import test_gaze_listen as g   # noqa: E402

checks = 0


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def new_store(tmp, name):
    persona = os.path.join(tmp, name + "_persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    return persona_store.Store(data_dir=os.path.join(tmp, name), persona_path=persona)


def wrote(st, name, impression="", note=""):
    """t20: one exchange, then the memory call's answer (impression / memo) stored as the app does."""
    st.after_reply(name, "안녕", "{}")
    p = st.impression(name)
    return st.memory_finish({"name": name, "impression": p["note"], "profile": p["profile"]},
                            {"changed": True, "profile": [], "impression": impression, "notes": [note] if note else []})


def store_file_checks(tmp):
    # ---- a JSON saved by Notepad as "UTF-8 with BOM" is read, not treated as empty
    folder = os.path.join(tmp, "bom", "npcs", "kwonjun")
    os.makedirs(folder)
    with open(os.path.join(folder, "impressions.json"), "w", encoding="utf-8-sig") as f:
        json.dump({"players": {"앨리스": {"note": "다정함", "locked": True, "updated": "2026-11-19 10:00", "talks": 3}}},
                  f, ensure_ascii=False)
    st = new_store(tmp, "bom")
    ok(st.players["앨리스"]["note"] == "다정함" and not st.load_problems, "BOM impressions.json is read")
    wrote(st, "밥", impression="처음 봄")
    again = new_store(tmp, "bom")
    ok(set(again.players) == {"앨리스", "밥"} and again.players["앨리스"]["locked"], "and survives the next save")

    # ---- a file that cannot be parsed is copied aside before anything can overwrite it
    folder = os.path.join(tmp, "bad", "npcs", "kwonjun")
    os.makedirs(folder)
    broken = '{"notes": [{"text": "소중한 메모"},'
    with open(os.path.join(folder, "memory.json"), "w", encoding="utf-8") as f:
        f.write(broken)
    st = new_store(tmp, "bad")
    ok(st.notes == [] and [p[0] for p in st.load_problems] == ["memory.json"], "broken memory.json starts empty")
    backup = st.load_problems[0][1]
    ok(backup.startswith("memory.json.broken-") and backup.endswith(".bak"), "the backup is named after the file")
    ok(st.add_note("새 메모") == "", "a new memo can still be written")
    with open(os.path.join(folder, backup), encoding="utf-8") as f:
        ok(f.read() == broken, "the original text is kept in the backup")
    ok(len([n for n in os.listdir(folder) if ".broken-" in n]) == 1, "one backup, not one per read")
    with open(os.path.join(folder, "settings.json"), "w", encoding="utf-8") as f:
        f.write("[1, 2]")                       # valid JSON of the wrong shape counts as unusable too
    ok([p[0] for p in new_store(tmp, "bad").load_problems] == ["settings.json"], "wrong-shaped JSON is backed up")
    ok(not new_store(tmp, "fresh").load_problems, "missing files are not a problem")


def capacity_checks(tmp):
    st = new_store(tmp, "full")
    st.apply({"notes": [{"text": f"내 메모 {i}", "by": "user"} for i in range(persona_store.MAX_NOTES)]})
    changed = wrote(st, "앨리스", note="AI가 적으려는 것")
    ok(len(st.notes) == persona_store.MAX_NOTES and all(n["by"] == "user" for n in st.notes)
       and changed["notes"] == [], "60 notes of the user: the AI's note is skipped, none is dropped")
    st.apply({"notes": [{"text": "AI 옛 메모", "by": "ai"}]
              + [{"text": f"내 메모 {i}", "by": "user"} for i in range(persona_store.MAX_NOTES - 1)]})
    changed = wrote(st, "앨리스", note="AI 새 메모")
    texts = [n["text"] for n in st.notes]
    ok(changed["notes"] == ["AI 새 메모"] and "AI 옛 메모" not in texts and texts[-1] == "AI 새 메모"
       and len(texts) == persona_store.MAX_NOTES, "an old AI note still makes room for a new one")

    st = new_store(tmp, "locked")
    st.apply({"players": {f"p{i}": {"note": "잠금", "locked": True} for i in range(persona_store.MAX_PLAYERS)}})
    changed = wrote(st, "새사람", impression="처음 봄")
    ok(len(st.players) == persona_store.MAX_PLAYERS and "새사람" not in st.players and changed["impression"] is None,
       "50 locked impressions: none is dropped for a newcomer")
    st.apply({"players": dict({f"p{i}": {"note": "잠금", "locked": True} for i in range(persona_store.MAX_PLAYERS - 1)},
                              열림={"note": "안 잠금", "locked": False})})
    wrote(st, "새사람", impression="처음 봄")
    ok("새사람" in st.players and "열림" not in st.players and len(st.players) == persona_store.MAX_PLAYERS,
       "an unlocked impression is the one that goes")


def merge_checks(tmp):
    st = new_store(tmp, "merge")
    st.apply({"notes": [{"text": "하나"}, {"text": "둘"}]})
    base = [n["text"] for n in st.notes]                     # the page loads here ...
    wrote(st, "앨리스", note="NPC가 그 사이에 적음")
    st.apply({"notes": [{"text": "하나 고침", "by": "user"}], "notes_base": base})   # ... edits one, deletes one
    ok([n["text"] for n in st.notes] == ["하나 고침", "NPC가 그 사이에 적음"],
       "a stale page keeps the memo the NPC added meanwhile")
    ok(st.notes[1]["by"] == "ai", "and its author")
    st.apply({"notes": [{"text": "통째로"}]})
    ok([n["text"] for n in st.notes] == ["통째로"], "no notes_base = the old whole-list replace")
    st.apply({"notes": [{"text": f"m{i}"} for i in range(persona_store.MAX_NOTES - 1)]})
    base = [n["text"] for n in st.notes]
    wrote(st, "앨리스", note="하나 더")
    before = [n["text"] for n in st.notes]
    try:
        st.apply({"notes": [{"text": t} for t in base] + [{"text": "넘침"}], "notes_base": base})
        ok(False, "an overflowing merge must be refused")
    except ValueError:
        ok([n["text"] for n in st.notes] == before, "an overflowing merge changes nothing")

    st = new_store(tmp, "merge_p")
    st.apply({"players": {"앨리스": {"note": "다정함", "locked": False}, "밥": {"note": "조용함", "locked": False}}})
    base = {k: dict(v) for k, v in st.players.items()}
    wrote(st, "앨리스", impression="야경을 좋아함") # NPC rewrites Alice
    wrote(st, "캐럴", impression="처음 봄") # and meets Carol
    st.set_locked("앨리스", True)
    page = {"앨리스": dict(base["앨리스"]), "밥": {"note": "조용하지만 웃김", "locked": False}}
    st.apply({"players": page, "players_base": base})
    ok(st.players["앨리스"]["note"] == "야경을 좋아함" and st.players["앨리스"]["locked"],
       "an impression the user did not touch keeps the NPC's newer text and lock")
    ok(st.players["밥"]["note"] == "조용하지만 웃김", "the edited one is saved")
    ok(st.players["캐럴"]["note"] == "처음 봄", "a player met meanwhile is kept")
    st.apply({"players": {"밥": {"note": "x", "locked": False}}, "players_base": {k: dict(v) for k, v in st.players.items()}})
    ok(set(st.players) == {"밥"}, "players removed on the page are removed")


def undo_checks(tmp):
    st = new_store(tmp, "undo")
    old = [{"text": f"옛 메모 {i}", "by": "ai", "at": "2026-11-19 10:00"} for i in range(persona_store.MAX_NOTES)]
    st.apply({"notes": [{"text": "요약", "by": "ai"}, {"text": "정리 뒤에 내가 적음", "by": "user"}]})
    data = {"backups": [{"at": "2026-11-19 20:10", "notes": old, "after": ["요약"], "players": {}}]}
    with open(os.path.join(st.data_dir, persona_store.COMPACT_FILE), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    why = st.compaction_undo()
    ok("61" in why and [n["text"] for n in st.notes] == ["요약", "정리 뒤에 내가 적음"],
       f"undo that would overflow is refused and changes nothing ({why})")
    ok(len(st.compaction_state()["backups"]) == 1, "the backup is still there")
    st.apply({"notes": [{"text": "요약", "by": "ai"}]})
    ok(st.compaction_undo() == "" and len(st.notes) == persona_store.MAX_NOTES, "undo works once it fits")


def clamp_checks():
    steps = persona_ai.clamp_steps([{"do": "look_at", "target": ["tv"]}, {"do": "look_at", "target": {"a": 1}}])
    ok([s["target"] for s in steps] == ["speaker", "speaker"], "a list / object target does not crash")
    ok(persona_ai._clamp_reply({"lines": "그냥 글자"})["line"] == "", "lines as a string does not crash")
    ok(persona_ai._clamp_reply({"lines": ["a"]})["line"] == "", "lines as a list does not crash")


def cfg_size_checks(tmp):
    st = new_store(tmp, "cfg")
    bridge = cfg_bridge.CfgBridge(st, set(), log=lambda *a, **k: None)
    st.apply({"notes": [{"text": f"{i:02d}" + "😀" * 78} for i in range(persona_store.MAX_NOTES)]})
    raw = bridge.payload("kwonjun")
    body = json.loads(raw)
    ok(len(raw) <= cfg_bridge.MAX_CFG_BYTES and body.get("notes_cut") is True and 0 < len(body["notes"]) < 60,
       f"/cfg stays under the byte limit ({len(raw)})")
    err = bridge.apply({"op": "notes_set", "text": "하나만", "expect": ""}, "a")
    ok(err and len(st.notes) == persona_store.MAX_NOTES, "a partly shown memo list cannot be saved back")
    st.apply({"notes": [{"text": "짧은 메모"}]})
    ok("notes_cut" not in json.loads(bridge.payload("kwonjun")), "a normal list is whole")
    ok(bridge.apply({"op": "notes_set", "text": "짧은 메모\n둘째", "expect": "짧은 메모"}, "a") == ""
       and len(st.notes) == 2, "and can be saved again")


def http_checks(tmp):
    st = new_store(tmp, "http")
    server = app.make_server(0, st, None, token="tok")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    old_wait, app.POLL_WAIT_S = app.POLL_WAIT_S, 0.2

    def get(path, **headers):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        out = res.status, res.read().decode("utf-8")
        conn.close()
        return out

    try:
        g_clear()
        app.replies.put({"line": "비밀 대답", "seq": 1})
        ok(get("/poll", Host="evil.example:%d" % port)[0] == 403 and app.replies.qsize() == 1,
           "/poll with another site's Host: refused, the reply stays")
        ok(get("/poll", Origin="http://evil.example")[0] == 403 and app.replies.qsize() == 1,
           "/poll from a web page (Origin): refused, the reply stays")
        ok(get("/poll", **{"Sec-Fetch-Site": "cross-site"})[0] == 403 and app.replies.qsize() == 1,
           "/poll as a cross-site browser request: refused")
        ok("비밀" not in get("/pollution")[1] and "비밀" not in get("/poll/x")[1] and app.replies.qsize() == 1,
           "only the exact path /poll takes a reply")
        ok(get("/ping", Host="evil.example")[0] == 403, "/ping checks Host too")
        ok(get("/ping")[1].startswith("pong") and get("/ping/burst1")[1].startswith("pong")
           and get("/ping", Host=f"localhost:{port}")[0] == 200, "/ping from this PC still answers")
        status, body = get("/poll?x=1")
        ok(status == 200 and json.loads(body)["line"] == "비밀 대답" and app.replies.empty(), "the world's own /poll gets it")
        ok(get("/poll") == (200, "{}"), "an empty poll still answers {}")
    finally:
        app.POLL_WAIT_S = old_wait
        server.shutdown()
        server.server_close()


def g_clear():
    while not app.replies.empty():
        app.replies.get_nowait()


class NamedBrain(g.SlowBrain):
    def __init__(self, store):
        super().__init__(store)
        self.speakers = []

    def reply(self, text, **kw):
        self.speakers.append(kw.get("speaker"))
        return super().reply(text, **kw)


def job_binding_checks(tmp):
    """Two clients are watched; the utterance belongs to the client whose log switched listening on."""
    persona = os.path.join(tmp, "bind_persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    store = persona_store.Store(data_dir=os.path.join(tmp, "bind_data"), persona_path=persona)
    bridge = cfg_bridge.CfgBridge(store, set(), log=app.log)
    brain, tr, mic = NamedBrain(store), g.FakeTr(), g.FakeMic()
    logs = os.path.join(tmp, "bind_vrc")
    os.makedirs(logs)
    a = os.path.join(logs, "output_log_2026-11-19_13-44-33.txt")
    b = os.path.join(logs, "output_log_2026-11-19_13-45-10.txt")
    for path, who in ((a, "Alice A"), (b, "Bob B")):
        with open(path, "w", encoding="utf-8") as f:
            f.write(g.join(g.LOCAL) + g.T + f'[Behaviour] Initialized PlayerAPI "{who}" is local\n')
    g_clear()
    stop = threading.Event()
    watcher = threading.Thread(target=app.watch_log, args=(logs, mic, tr, False, brain, store, bridge, stop),
                               kwargs={"listen_mode": "gaze"}, daemon=True)
    watcher.start()
    try:
        g.wait_for(lambda: store.speaker_known, "backlog read")
        time.sleep(0.3)
        g.add(b, g.T + '[Behaviour] Initialized PlayerAPI "Bob B" is local\n')    # B is "whoever joined last"
        g.wait_for(lambda: store.speaker == "Bob B", "global speaker is Bob")
        g.add(a, g.LISTEN_ON)
        g.wait_for(lambda: mic.tap is not None, "listening for client A")
        tr.text = "나는 앨리스야"
        mic.push(np.concatenate([g.room(0.6), g.speech(1.2), g.room(1.4)]))
        reply = app.replies.get(timeout=5)
        ok(reply["heard"] == "나는 앨리스야" and brain.speakers == ["Alice A"],
           f"the utterance carries client A's player, not the last one to join ({brain.speakers})")

        # ---- a sentence that started as ordinary talk stays talk when a dictation is armed before it ends
        g.wait_for(lambda: len(g.tsv("REPLY")) >= 1, "first turn done")
        time.sleep(0.2)
        tr.text = "이건 그냥 대화"
        mic.push(np.concatenate([g.room(0.3), g.speech(1.0)]))
        g.wait_for(mic.drained, "first half heard")
        time.sleep(0.2)
        ok(bridge.apply({"op": "dictate", "target": "note_add"}, a) == "", "dictation armed mid-sentence")
        mic.push(np.concatenate([g.speech(0.5), g.room(1.4)]))
        reply = app.replies.get(timeout=5)
        ok(reply["heard"] == "이건 그냥 대화" and brain.heard[-1] == "이건 그냥 대화" and not store.notes
           and bridge.dictate_target() is not None, "it is answered as talk and the dictation keeps waiting")

        # ---- a sentence spoken FOR the panel is thrown away when that dictation is cancelled meanwhile
        g.wait_for(lambda: len(g.tsv("REPLY")) >= 2, "second turn done")
        time.sleep(0.2)
        heard = len(brain.heard)
        tr.text = "받아쓰기용 말"
        mic.push(np.concatenate([g.room(0.3), g.speech(1.0)]))
        g.wait_for(mic.drained, "dictation speech started")
        time.sleep(0.2)
        ok(bridge.apply({"op": "dictate", "target": "none"}, a) == "", "dictation cancelled mid-sentence")
        mic.push(np.concatenate([g.speech(0.5), g.room(1.4)]))
        g.wait_for(lambda: any("dictate_gone" in line for line in g.tsv("STT")), "the job was dropped")
        ok(g.no_reply() and len(brain.heard) == heard and not store.notes,
           "a cancelled dictation is neither stored nor sent to the AI")

        # ---- and the normal case still works: armed before the sentence, stored as a memo
        ok(bridge.apply({"op": "dictate", "target": "note_add"}, a) == "", "dictation armed")
        tr.text = "귤을 좋아한다"
        mic.push(np.concatenate([g.room(0.6), g.speech(1.2), g.room(1.4)]))
        g.wait_for(lambda: [n["text"] for n in store.notes] == ["귤을 좋아한다"], "dictated memo stored")
        ok(len(brain.heard) == heard and bridge.dictate_target() is None, "dictation goes to the memo, not to the AI")
    finally:
        stop.set()
        watcher.join(timeout=3)


def main():
    tmp = tempfile.mkdtemp(prefix="w04_triage_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")        # keep the real log clean
        store_file_checks(tmp)
        capacity_checks(tmp)
        merge_checks(tmp)
        undo_checks(tmp)
        clamp_checks()
        cfg_size_checks(tmp)
        http_checks(tmp)
        job_binding_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_regression_checks():
    main()


if __name__ == "__main__":
    main()
