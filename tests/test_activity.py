"""Offline check of the NPC activity record (activity.py, act v1; TODO 118 steps 1).

    .venv\\Scripts\\python.exe -X utf8 tests\\test_activity.py

No microphone, no Whisper, no API call, no real VRChat log: a fake log in a temporary folder, the real
watcher loop and HTTP server on a spare port, a scripted brain. The format itself (rows, 1500/7000 bytes)
is checked with the shared examples in tests/test_golden_contract.py.
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

import activity                # noqa: E402
import cfg_bridge              # noqa: E402
import persona_ai              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402

NS = types.SimpleNamespace
checks = 0
LOCAL = "local:1a7a51216f56f32584745f9ce2887998a0e3a76e83f7e20749da576bb20137da"
T = "2026.11.19 13:44:37 Debug      -  "
CFG_OPEN = T + "[W04AI] CFG_OPEN utc=639306602829378126\n"
PTT = T + "[W04AI] PTT_DOWN utc=1\n" + T + "[W04AI] PTT_UP held_ms=1500 utc=2\n"


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def get(port, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("GET", path)
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
    text = ""

    def run(self, audio):
        return {"text": self.text, "segments": []}


def base(line):
    return {"emotion": "happy", "intensity": 0.5, "steps": [], "lines": {"ko": line}, "line": line,
            "ai_ms": 1200, "model": "mock"}


class ScriptedBrain:
    """Stands in for persona_ai.Brain: each reply() takes the next scripted turn."""

    def __init__(self, store):
        self.store, self.model, self.turns, self.heard, self.memory, self.kw = store, "mock", [], [], [], []

    def reply(self, text, on_interim=None, **kw):
        self.heard.append(text)
        self.kw.append(kw)
        turn = self.turns.pop(0)
        if turn.get("interim") and on_interim is not None:
            on_interim(turn["interim"])
            time.sleep(0.3)
        if turn.get("memory"):      # like Brain.reply: the exchange is stored, then the app's memory call reads it
            self.store.after_reply(kw["speaker"], text, json.dumps({"lines": {"ko": turn["out"]["line"]}}),
                                   utterance={"speaker": kw["speaker"], "source": "PC 주인 마이크", "time": "", "raw": text})
            self.memory.append(turn["memory"])
        return turn["out"]

    def update_memory(self, job):
        """The memory call (t20): the scripted answer, else "nothing changed"."""
        if self.memory:
            return self.memory.pop(0), {"in": 100, "out": 20}
        return {"changed": False, "profile": job["profile"], "impression": job["impression"], "notes": []}, {}


def call(kind, ms):
    return {"kind": kind, "ms": ms, "cache_read": 0, "cache_write": 0, "fp_system": "-", "fp_schema": "-",
            "thinking": "disabled", "effort": "low"}


def kinds(act):
    return [row["k"] for row in act["rows"]]


def brain_checks(tmp):
    """persona_ai.Brain reports what the record needs: dropped-step notes, map rows, the interim query."""
    persona = os.path.join(tmp, "persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    store = persona_store.Store(data_dir=os.path.join(tmp, "brain"), persona_path=persona)
    store.set_speaker("Alice")
    brain = persona_ai.Brain("fake", store)
    answers = []

    def create(**kw):
        item = answers.pop(0)
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(item, ensure_ascii=False))],
                  usage=NS(input_tokens=10, output_tokens=5))

    brain.client, brain.mode = NS(messages=NS(create=create)), "haiku"
    answers.append({"emotion": "happy", "intensity": 0.5, "lines": {"ko": "소파에 앉을게."},
                    "steps": [{"do": "go_sit"}, {"do": "outfit", "target": "목리본", "action": "off"}]})
    r = brain.reply("셔츠 벗고 소파에 앉아")
    ok(r["step_notes"] == ["outfit"] and "옷 메뉴에 없는 건" in r["line"],
       "the note the player hears is reported as step_notes (the same source)")
    ok(r.get("map_used") and "소파" in r["map_used"], "map rows the reply used are named")
    notes = []
    persona_ai._clamp_reply({"lines": {"ko": "좋아."}, "steps": [{"do": "keep_distance", "action": "20"}]},
                            notes_out=notes)
    ok(notes == ["distance_min"], "_clamp_reply hands out its notes")
    notes = []
    persona_ai._clamp_reply({"lines": {"ko": "좋아."}, "choices": ["가", "나"],
                             "steps": [{"do": "outfit", "target": "x", "action": "off"}]}, notes_out=notes)
    ok(notes == [], "a note that was never said (choices) is not reported")


def watcher_checks(tmp):
    persona = os.path.join(tmp, "persona.md")
    store = persona_store.Store(data_dir=os.path.join(tmp, "data"), persona_path=persona)
    bridge = cfg_bridge.CfgBridge(store, set(), log=app.log)
    brain, tr = ScriptedBrain(store), FakeTr()
    logs = os.path.join(tmp, "vrc")
    os.makedirs(logs)
    log_a = os.path.join(logs, "output_log_2026-11-19_13-44-33.txt")
    with open(log_a, "w", encoding="utf-8") as f:
        f.write(T + "[Behaviour] Joining " + LOCAL + "\n" + T + '[Behaviour] Initialized PlayerAPI "Tester One" is local\n')

    def add(text):
        with open(log_a, "a", encoding="utf-8") as f:
            f.write(text)

    def poll():
        """The next body, past the hearing / thinking statuses (those carry no act)."""
        while True:
            status, raw = get(port, "/poll")
            ok(status == 200, "poll answered")
            body = json.loads(raw)
            if body.get("state") not in ("hearing", "thinking"):
                return raw, body
            ok("act" not in body, "a plain status has no act")

    while not app.replies.empty():
        app.replies.get_nowait()
    activity.journal.clear()
    stop = threading.Event()
    watcher = threading.Thread(target=app.watch_log, args=(logs, FakeRec(), tr, False, brain, store, bridge, stop),
                               daemon=True)
    watcher.start()
    server = app.make_server(0, store, brain, bridge=bridge)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg_bridge.OPEN_GRACE_S = 0.0
    try:
        wait_for(lambda: store.speaker == "Tester One", "watcher read the backlog")
        time.sleep(0.3)

        # ---- an ordinary turn: heard -> think -> map / dropped step -> reply, in one body
        out = base("소파에 앉을게. 옷 메뉴에 없는 건 빼고 할게.")
        out.update(step_notes=["outfit"], map_used=["소파", "TV"], calls=[call("first", 900)])
        brain.turns.append({"out": out})
        tr.text = "셔츠 벗고 소파에 앉아"
        add(PTT)
        raw, body = poll()
        act = body.get("act")
        ok(act and activity.check(act) and act["npc"] == "kwonjun" and act["turn"] == body["seq"],
           "the reply carries a valid act for its own turn")
        ok(kinds(act) == ["heard", "think", "map", "drop", "reply"], f"rows in order: {kinds(act)}")
        ok(act["rows"][0]["x"] == "셔츠 벗고 소파에 앉아" and "ms" in act["rows"][0], "heard row: the transcript and STT time")
        ok(act["rows"][3]["x"] == "옷 메뉴에 없는 건 빼고 할게." and act["rows"][2]["x"] == "소파 · TV",
           "drop row says what the player heard; map row names the rows")
        ok(act["rows"][-1] == {"d": act["rows"][-1]["d"], "k": "reply", "x": body["line"], "ms": 1200},
           "reply row: the line and the AI time")
        ok(all(a["d"] <= b["d"] for a, b in zip(act["rows"], act["rows"][1:])), "d never goes backwards")
        ok(body["line"] == out["line"] and body["heard"] == "셔츠 벗고 소파에 앉아", "the reply itself is as before")

        # ---- a search turn: two bodies with the same turn number
        interim = base("잠깐만, 찾아볼게!")
        interim.update(interim="search", search_query="서울 날씨")
        final = base("오늘 서울은 맑고 12도래.")
        final.update(search_query="서울 날씨", links=[{"title": "기상청 날씨누리", "url": "https://x"}],
                     calls=[call("first", 800), call("search", 4000)])
        brain.turns.append({"interim": interim, "out": final})
        tr.text = "서울 날씨 알려 줘"
        add(PTT)
        _, first = poll()
        ok(first["state"] == "searching" and kinds(first["act"]) == ["heard", "think", "search"]
           and first["act"]["rows"][2]["x"] == "서울 날씨", "the searching status already shows the query")
        _, second = poll()
        ok(second["line"] == final["line"] and second["act"]["turn"] == first["act"]["turn"],
           "the answer comes with the same turn number")
        ok(kinds(second["act"]) == ["heard", "think", "search", "source", "reply"]
           and second["act"]["rows"][3]["x"] == "기상청 날씨누리", f"search rows once, then the source: {kinds(second['act'])}")

        app.replies.protected_until.clear()     # the search answer's play window would hold the next one
        # ---- a recall turn: the interim body is a real reply; ms of the recall comes with the answer
        interim = base("잠깐 생각해 볼게.")
        interim.update(interim="recall")
        final = base("토요일이었어!")
        final.update(recall=True, recall_ms=900, calls=[call("first", 500), call("recall", 900)])
        memory = {"changed": True, "profile": ["토요일 약속을 확인함"], "impression": "약속을 잘 챙김",
                  "notes": ["Tester One과 토요일에 영화 보기로 함"]}
        brain.turns.append({"interim": interim, "out": final, "memory": memory})
        tr.text = "저번 약속 기억나?"
        add(PTT)
        _, first = poll()
        ok(first["line"] == "잠깐 생각해 볼게." and kinds(first["act"]) == ["heard", "think", "recall"]
           and "ms" not in first["act"]["rows"][2], "the waiting line shows the recall started")
        _, second = poll()
        recall = [row for row in second["act"]["rows"] if row["k"] == "recall"]
        ok(len(recall) == 1 and recall[0]["ms"] == 900, "one recall row, now with its time")
        ok("memory" not in kinds(second["act"]) and "impression" not in kinds(second["act"]),
           "t20: the reply itself never carries memory rows (the memory call runs after it)")

        # ---- t20: the memory call after that reply -> /cfg at once, and the front of the next act
        wait_for(lambda: [row["k"] for row in activity.journal.recent("kwonjun")["rows"]][-2:] == ["impression", "memory"],
                 "the memory call finished after the reply")
        ok(store.players["Tester One"]["note"] == "약속을 잘 챙김"
           and store.players["Tester One"]["profile"] == ["토요일 약속을 확인함"]
           and store.notes[-1]["text"] == "Tester One과 토요일에 영화 보기로 함", "stored by the memory call")
        with open(app.LOG_PATH, encoding="utf-8") as f:
            logged = f.read()
        ok("\tIMPRESSION\tnpc=kwonjun\tTester One\t약속을 잘 챙김" in logged
           and "\tPROFILE\tnpc=kwonjun\tTester One\t토요일 약속을 확인함" in logged
           and "\tMEMORY\tnpc=kwonjun\tTester One과 토요일에 영화 보기로 함" in logged
           and "\tMEMORY_CALL\tnpc=kwonjun\twho=Tester One\tok=1\tchanged=1" in logged, "log lines with their text")

        # ---- an error turn: a plain label, never the exception text
        failed = base("음… 미안, 잠깐 멍했어.")
        failed.update(error="RuntimeError: secret detail from the API")
        brain.turns.append({"out": failed})
        tr.text = "안녕"
        add(PTT)
        _, body = poll()
        ok("error" in kinds(body["act"]) and "secret" not in json.dumps(body["act"], ensure_ascii=False),
           "error row is a fixed label")
        ok(kinds(body["act"])[:3] == ["impression", "memory", "heard"]
           and body["act"]["rows"][0] == {"d": 0, "k": "impression", "x": "Tester One(나): 약속을 잘 챙김"},
           f"the memory call's rows ride at the front of the next act, once ({kinds(body['act'])})")

        # ---- overheard (for_me false): the push-to-talk still gets its empty answer, with a skip row
        brain.turns.append({"out": dict(base(""), for_me=False)})
        tr.text = "저기 너 말고"
        add(PTT)
        _, body = poll()
        ok(body["line"] == "" and kinds(body["act"]) == ["heard", "think", "skip"], "skip is recorded")

        # ---- t20: a visitor's text chat (CHAT_REQ) is answered like speech, for that visitor
        brain.turns.append({"out": base("응, 반가워!")})
        add(T + "[W04AI] CHAT_REQ npc=kwonjun id=77 by=Visitor Two q=  안녕   키보드로 말해  \n")
        _, body = poll()
        ok(body["line"] == "응, 반가워!" and body["req"] == 77 and body["by"] == "Visitor Two"
           and body["heard"] == "안녕 키보드로 말해", "the chat reply echoes the request number and the visitor")
        sent = brain.kw[-1]
        with open(app.LOG_PATH, encoding="utf-8") as f:
            logged = f.read()
        ok(sent["speaker"] == "Visitor Two" and "\tCHAT_REQ\tseq=" in logged and "\treq=77\tby=Visitor Two\tlen=10" in logged,
           "speaker = the visitor (the real Brain also gets source 월드 글 채팅 and no not-for-me filter)")
        ok(kinds(body["act"])[0] == "heard" and "ms" not in body["act"]["rows"][0], "a typed line has no STT time")
        brain.turns.append({"out": base("또 왔네")})
        add(T + "[W04AI] CHAT_REQ npc=kwonjun by=Visitor Two q=두 번째\n"
            + T + "[W04AI] CHAT_REQ npc=kwonjun id=5 by=Nobody q=   \n")
        _, body = poll()
        ok(body["line"] == "또 왔네" and body["req"] == body["seq"] and body["by"] == "Visitor Two",
           "no id= -> the app's own turn number; an empty text is ignored")

        # ---- a body that would pass 7000 bytes: the record gives way, the reply never
        huge = base("가" * 1130)
        brain.turns.append({"out": huge})
        tr.text = "긴 이야기 해 줘"
        add(PTT)
        raw, body = poll()
        ok(body["line"] == huge["line"] and len(raw) <= activity.MAX_BODY_BYTES,
           f"reply intact and the body within 7000 bytes ({len(raw)})")
        ok("act" not in body or len(body["act"]["rows"]) < 3, "old rows (or the whole act) left out")
        brain.turns.append({"out": base("가" * 1200)})
        add(PTT)
        raw, body = poll()
        ok("act" not in body and body["line"] == "가" * 1200, "a reply over 7000 bytes alone goes out without act")

        # ---- dictation: no record at all
        before = activity.journal.recent("kwonjun")["rows"]
        bridge.apply({"op": "dictate", "target": "note_add"})
        tr.text = "루나는 딸기를 좋아한다"
        add(PTT)
        raw, body = poll()
        ok(body.get("dictate") == "note_add" and "act" not in body, "a dictation answer has no act")
        ok(activity.journal.recent("kwonjun")["rows"] == before and "딸기".encode() not in raw,
           "the dictated text is not in the record")

        # ---- /cfg: the last 20 rows, with the PC's clock instead of d
        add(CFG_OPEN)
        wait_for(bridge.is_open, "window opened")
        status, raw = get(port, "/cfg")
        cfg = json.loads(raw)
        rows = cfg["act"]["rows"]
        ok(status == 200 and cfg["act"]["v"] == 1 and cfg["act"]["npc"] == "kwonjun"
           and len(rows) == activity.HISTORY_ROWS, f"/cfg carries the last 20 rows ({len(rows)})")
        ok(all(set(row) <= {"t", "k", "x", "ms", "c"} and len(row["t"]) == 8 for row in rows), "cfg rows: t, k, x, ms, c")
        ok(any(row.get("c") for row in rows) and isinstance(cfg["errors"], list), "/cfg rows carry codes (TODO 140)")
        ok(rows[-1]["k"] == "reply", "newest last")
        ok(json.loads(get(port, "/cfg?npc=itoa")[1])["act"]["rows"] == [], "each NPC has its own record")
        person = cfg["memory"]["people"][0]
        ok(person["name"] == "Tester One" and person["note"] == "약속을 잘 챙김" and person["before"] == ""
           and person["profile"] == ["토요일 약속을 확인함"] and cfg["memory"]["changes"] == 1
           and "U\tTester One\t약속을 잘 챙김\t토요일 약속을 확인함" in cfg["memory"]["digest"],
           "/cfg memory: the person, what changed and the Global digest")
    finally:
        cfg_bridge.OPEN_GRACE_S = 2.0
        stop.set()
        server.shutdown()
        server.server_close()
        watcher.join(timeout=3)


def unit_checks(tmp):
    plain = {"line": "응", "seq": 1}
    ok(app.poll_output(plain) is plain, "a reply without act goes out exactly as before")
    reply = app.TimedReply({"line": "응", "seq": 1, "npc": "kwonjun",
                            "act": activity.pack("kwonjun", 1, [{"d": 0, "k": "think"}])},
                           NS(tail=None, npc_epoch=0))
    out = app.poll_output(reply)
    ok(out["act"]["rows"] == [{"d": 0, "k": "think", "x": ""}] and list(out)[-1] == "act", "act rides at the end")
    # the cfg answer: the record gives way before talk history and memos
    persona = os.path.join(tmp, "persona.md")
    store = persona_store.Store(data_dir=os.path.join(tmp, "cfg"), persona_path=persona)
    bridge = cfg_bridge.CfgBridge(store)
    activity.journal.clear()
    for i in range(30):
        activity.journal.add("kwonjun", {"k": "reply", "x": "가" * 60, "t": "14:00:00"})
    store.apply({"notes": [{"text": "나" * 59} for _ in range(60)]})
    body = json.loads(bridge.payload("kwonjun"))
    ok(len(body["act"]["rows"]) < activity.HISTORY_ROWS and len(body["notes"]) == 60
       and len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()) <= cfg_bridge.MAX_CFG_BYTES,
       "a full /cfg drops old record rows before memos")
    activity.journal.clear()


def main():
    tmp = tempfile.mkdtemp(prefix="w04_activity_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")        # keep the real log clean
        shutil.copy(os.path.join(ROOT, "persona.md"), os.path.join(tmp, "persona.md"))
        brain_checks(tmp)
        unit_checks(tmp)
        watcher_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_activity_checks():
    main()


if __name__ == "__main__":
    main()
