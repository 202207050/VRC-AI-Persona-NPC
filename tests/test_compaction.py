"""Offline check of automatic memory compaction (no API call, no network, no real data).

    .venv\\Scripts\\python.exe -X utf8 tests\\test_compaction.py

The model is replaced by a fake brain / fake client; all files go to a temporary folder.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)

import cfg_bridge              # noqa: E402
import persona_ai              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402
from voice_options import MODELS  # noqa: E402

NS = types.SimpleNamespace
checks = 0


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


class FakeBrain:
    """Stands in for persona_ai.Brain.compact_memory: records the job, returns the queued answer."""

    def __init__(self):
        self.jobs, self.answers, self.gate = [], [], None

    def compact_memory(self, job):
        self.jobs.append(job)
        if self.gate is not None:
            self.gate.wait(5)
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {"in": 4000, "out": 2000}


def new_store(tmp, name="data"):
    persona = os.path.join(tmp, "persona.md")
    if not os.path.exists(persona):
        shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    return persona_store.Store(data_dir=os.path.join(tmp, name), persona_path=persona)


def fill(store, ai=52, user=("손님은 커피를 싫어한다", "토요일에 같이 영화 보기로 약속함"), tag="기억"):
    notes = [{"text": t, "by": "user"} for t in user]
    notes += [{"text": f"{tag} {i:02d}: 손님이 좋아하는 것 이야기", "by": "ai"} for i in range(ai)]
    store.apply({"notes": notes})


def run(store, brain, force=False):
    return app.start_compaction(store, brain, force=force, wait=True)


def main():
    tmp = tempfile.mkdtemp(prefix="w04_compact_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")
        store, brain = new_store(tmp), FakeBrain()
        s = store.settings
        ok(s["compact_on"] is True and s["compact_note_count"] == 50 and s["compact_note_chars"] == 3000
           and s["compact_impression_chars"] == 400 and s["compact_user_notes"] is False and s["compact_minutes"] == 10,
           "defaults: on, >50 memos or >3,000 chars, impression >400 chars, user memos kept, once per 10 minutes")
        ok(persona_store.MAX_NOTES > 50 and persona_store.MAX_IMPRESSION_CHARS > 400, "caps leave room above the thresholds")

        # ---- thresholds
        fill(store, ai=20)
        ok(run(store, brain) != "" and not brain.jobs, "22 short memos: nothing to do, no model call")
        fill(store, ai=52)
        ok(len(store.notes) == 54, "54 memos stored")
        brain.answers.append({"notes": ["손님은 귤과 야경을 좋아한다", "손님은 공포 영화를 못 본다", "손님은 커피를 싫어한다"],
                              "impressions": []})
        ok(run(store, brain) == "", "more than 50 memos starts a compaction")
        job = brain.jobs[0]
        ok(len(job["notes"]) == 52 and all(n["by"] == "ai" for n in job["notes"]), "only the AI's memos are sent to be condensed")
        ok(job["keep"] == ["손님은 커피를 싫어한다", "토요일에 같이 영화 보기로 약속함"], "the user's memos go along as read-only context")
        texts = [n["text"] for n in store.notes]
        ok(texts == ["손님은 커피를 싫어한다", "토요일에 같이 영화 보기로 약속함", "손님은 귤과 야경을 좋아한다", "손님은 공포 영화를 못 본다"],
           "user memos kept word for word and first; a condensed memo that repeats one of them is dropped")
        ok([n["by"] for n in store.notes] == ["user", "user", "ai", "ai"], "authors kept")
        ok(json.load(open(os.path.join(store.data_dir, "memory.json"), encoding="utf-8"))["notes"][2]["text"]
           == "손님은 귤과 야경을 좋아한다", "written to memory.json")
        st = store.state()["compaction"]
        ok(st["last"]["ok"] and st["last"]["notes_before"]["count"] == 54 and st["last"]["notes_after"]["count"] == 4
           and st["last"]["notes_after"]["chars"] < st["last"]["notes_before"]["chars"], "before/after sizes recorded")
        ok(st["last"]["usage"] == {"in": 4000, "out": 2000} and abs(st["last"]["cost_usd"] - 0.0014) < 1e-9,
           "cost from the Haiku 5.5 prices ($0.10 in / $0.50 out per million tokens)")
        ok(len(st["backups"]) == 1 and st["running"] is False, "one backup")
        log = open(app.LOG_PATH, encoding="utf-8").read()
        ok("\tCOMPACT\tnpc=kwonjun\tok=1\tnotes=54>4\t" in log and "usd=0.001400" in log, "one short log line")
        ok("귤" not in log, "memo text is not written to the log")

        # ---- at most once per N minutes
        fill(store, ai=52)
        ok("기다리는" in run(store, brain) and len(brain.jobs) == 1, "a second run inside the interval is skipped")
        data_path = os.path.join(store.data_dir, persona_store.COMPACT_FILE)

        def rewind(minutes=11):
            data = json.load(open(data_path, encoding="utf-8"))
            data["last_try"] = time.time() - minutes * 60
            json.dump(data, open(data_path, "w", encoding="utf-8"), ensure_ascii=False)

        rewind(9)
        ok(run(store, brain) != "" and len(brain.jobs) == 1, "9 minutes later: still waiting")
        store.apply({"settings": {"compact_minutes": 5}})
        brain.answers.append({"notes": ["손님은 귤을 좋아한다"], "impressions": []})
        ok(run(store, brain) == "" and len(store.notes) == 3, "interval is a setting (5 minutes): runs again")
        store.apply({"settings": {"compact_minutes": 10}})

        # ---- undo, newest first; automatic compaction goes off
        ok(len(store.state()["compaction"]["backups"]) == 2, "two backups")
        store.add_note("정리 뒤에 내가 적은 메모")
        ok(store.compaction_undo() == "" and len(store.notes) == 55, "undo restores the 54 memos and keeps the one written since")
        ok(store.notes[-1]["text"] == "정리 뒤에 내가 적은 메모" and store.notes[2]["text"].startswith("기억 00"), "restored in order")
        ok(store.settings["compact_on"] is False
           and json.load(open(os.path.join(store.data_dir, "settings.json"), encoding="utf-8"))["compact_on"] is False,
           "undo switches automatic compaction off (saved)")
        rewind()
        ok("꺼져" in run(store, brain), "switched off: no automatic run")
        ok(store.compaction_undo() == "" and store.compaction_undo() != "", "second undo works, a third has nothing left")

        # ---- only the last 3 backups are kept
        store.apply({"settings": {"compact_on": True}})
        for i in range(4):
            fill(store, ai=52)
            rewind()
            brain.answers.append({"notes": [f"묶음 {i}"], "impressions": []})
            ok(run(store, brain) == "", f"run {i}")
        ok(len(store.state()["compaction"]["backups"]) == persona_store.COMPACT_BACKUPS == 3, "last 3 backups kept")

        # ---- bad answers never lose memos
        for tag, (bad, why) in enumerate((({"notes": [], "impressions": []}, "an empty answer"),
                         ({"notes": [f"기억 {i:02d}: 손님이 좋아하는 것 이야기 그리고 더 길게" for i in range(52)], "impressions": []},
                          "an answer that is not shorter"),
                         (RuntimeError("network"), "a failed call"))):
            fill(store, ai=52, tag=f"실패{tag}")
            rewind()
            brain.answers.append(bad)
            before = [dict(n) for n in store.notes]
            ok(run(store, brain) == "" and store.notes == before, why + " changes nothing")
            ok(store.state()["compaction"]["last"]["ok"] is False and len(store.state()["compaction"]["backups"]) == 3,
               why + ": recorded as not changed, no new backup")
        ok("\tok=0\t" in open(app.LOG_PATH, encoding="utf-8").read(), "failed runs are logged too")
        calls = len(brain.jobs)
        rewind()
        brain.answers.append({"notes": ["손님은 귤을 좋아한다"], "impressions": []})
        ok(run(store, brain) == "" and len(brain.jobs) == calls + 1 and len(store.notes) == 3,
           "after a failed call the same memos are tried again once the interval has passed")
        calls = len(brain.jobs)

        # ---- total characters threshold + "compress my memos too"
        store2 = new_store(tmp, "data2")
        store2.apply({"notes": [{"text": f"{i:02d} " + "가" * 76, "by": "user"} for i in range(40)]})   # 40 x 79 = 3,160 chars
        ok(run(store2, brain) != "" and len(brain.jobs) == calls, "only user memos and the switch is off: no call at all")
        store2.apply({"settings": {"compact_user_notes": True}})
        brain.answers.append({"notes": ["손님 이야기를 한 줄로 줄임"], "impressions": []})
        ok(run(store2, brain) == "" and [n["text"] for n in store2.notes] == ["손님 이야기를 한 줄로 줄임"],
           "over 3,000 characters with 'compress my memos too' on: user memos are condensed")
        ok(brain.jobs[-1]["keep"] == [] and len(brain.jobs[-1]["notes"]) == 40, "then they are sent as memos, not as keep")

        # ---- the same memos are not sent twice when nothing could be reduced
        store3 = new_store(tmp, "data3")
        fill(store3, ai=52)
        brain.answers.append({"notes": [], "impressions": []})
        run(store3, brain)
        n = len(brain.jobs)
        data_path3 = os.path.join(store3.data_dir, persona_store.COMPACT_FILE)
        data = json.load(open(data_path3, encoding="utf-8"))
        data["last_try"] = 0
        json.dump(data, open(data_path3, "w", encoding="utf-8"), ensure_ascii=False)
        ok(run(store3, brain) != "" and len(brain.jobs) == n, "unchanged memos are not sent again (no repeated cost)")

        # ---- impressions: over 400 characters, unlocked only
        store4 = new_store(tmp, "data4")
        long_a, long_b = ("다정함." * 120)[:450], ("조용함." * 120)[:450]
        store4.apply({"players": {"Alice": {"note": long_a}, "Bob": {"note": long_b, "locked": True},
                                  "Carol": {"note": "짧은 인상"}}})
        ok(len(store4.players["Alice"]["note"]) == 450, "a hand-written impression may be longer than 400 characters")
        brain.answers.append({"notes": [], "impressions": [{"name": "Alice", "note": "다정함. 야경과 귤을 좋아함."},
                                                            {"name": "Bob", "note": "바꾸면 안 됨"},
                                                            {"name": "Mallory", "note": "없는 사람"}]})
        ok(run(store4, brain) == "", "an impression over 400 characters starts a compaction")
        ok(brain.jobs[-1]["impressions"] == {"Alice": long_a.strip()} and brain.jobs[-1]["notes"] == [],
           "only the long unlocked impression is sent")
        ok(store4.players["Alice"]["note"] == "다정함. 야경과 귤을 좋아함." and store4.players["Bob"]["note"] == long_b.strip()
           and "Mallory" not in store4.players, "locked and unknown players are untouched")
        last = store4.state()["compaction"]["last"]
        ok(last["impressions"] == [{"name": "Alice", "before": len(long_a.strip()), "after": 16}], "impression sizes recorded")
        ok(store4.compaction_undo() == "" and store4.players["Alice"]["note"] == long_a.strip(), "undo restores the impression")

        # ---- runs in the background and never overwrites edits made meanwhile
        store5 = new_store(tmp, "data5")
        fill(store5, ai=52)
        slow = FakeBrain()
        slow.gate = threading.Event()
        slow.answers.append({"notes": ["손님은 귤을 좋아한다"], "impressions": []})
        t0 = time.time()
        ok(app.start_compaction(store5, slow) == "" and time.time() - t0 < 1.0, "start_compaction returns at once (background thread)")
        deadline = time.time() + 5
        while not slow.jobs and time.time() < deadline:
            time.sleep(0.01)
        ok(store5.state()["compaction"]["running"] is True, "shown as running")
        ok("이미" in app.start_compaction(store5, slow), "no second run while one is running")
        ok(store5.compaction_undo() != "", "undo waits while a run is in progress")
        store5.after_reply("Alice", "나 수요일에 이사해", "{}", memory_note="Alice는 수요일에 이사한다")   # the AI replies meanwhile
        store5.edit_note(2, "", None)                                                          # the user deletes one memo
        slow.gate.set()
        deadline = time.time() + 5
        while store5.state()["compaction"]["running"] and time.time() < deadline:
            time.sleep(0.01)
        texts = [n["text"] for n in store5.notes]
        ok(texts == ["손님은 커피를 싫어한다", "토요일에 같이 영화 보기로 약속함", "손님은 귤을 좋아한다", "Alice는 수요일에 이사한다"],
           "a memo added while the call ran is kept after the condensed ones")

        # ---- settings: validation, persistence
        for bad in ({"compact_note_count": 0}, {"compact_note_count": persona_store.MAX_NOTES}, {"compact_note_chars": "3000"},
                    {"compact_impression_chars": 500}, {"compact_minutes": 0}, {"compact_note_chars": 1.5}):
            try:
                store.apply({"settings": bad})
                ok(False, f"bad setting accepted: {bad}")
            except ValueError:
                ok(True, "bad setting refused")
        store.apply({"settings": {"compact_note_count": 30.0, "compact_note_chars": 2000, "compact_impression_chars": 300,
                                  "compact_minutes": 30, "compact_user_notes": True, "compact_on": False}})
        again = new_store(tmp)
        ok(again.settings["compact_note_count"] == 30 and again.settings["compact_note_chars"] == 2000
           and again.settings["compact_impression_chars"] == 300 and again.settings["compact_minutes"] == 30
           and again.settings["compact_user_notes"] is True and again.settings["compact_on"] is False,
           "thresholds survive a restart")
        ok(len(again.state()["compaction"]["backups"]) == 3, "backups survive a restart")

        # ---- the real request (fake client): Haiku, thinking off, small cap, structured output
        requests = []

        class Messages:
            stop = "end_turn"

            def with_options(self, **kw):
                requests.append(("options", kw))
                return NS(messages=self)

            def create(self, **kw):
                requests.append(("create", kw))
                text = json.dumps({"notes": ["손님은 귤을 좋아한다"], "impressions": []}, ensure_ascii=False)
                return NS(stop_reason=self.stop, content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
                          usage=NS(input_tokens=4321, output_tokens=987))

        real = persona_ai.Brain("fake", new_store(tmp, "data6"))
        messages = Messages()
        real.client = NS(messages=messages, with_options=messages.with_options)
        real.store.apply({"settings": {"brain_model": MODELS["opus"]}})
        job = {"npc": "kwonjun", "notes": [{"text": "손님은 귤을 좋아함", "by": "ai"}, {"text": "손님은 귤을 아주 좋아함", "by": "ai"}],
               "keep": ["내가 적은 메모"], "impressions": {"Alice": "다정함. " * 80}, "before": {"count": 3, "chars": 30}}
        result, usage = real.compact_memory(job)
        kw = requests[-1][1]
        ok(requests[0] == ("options", {"max_retries": 0}), "no automatic retry")
        ok(kw["model"] == MODELS["haiku"], "always Haiku, even when the NPC answers with Opus")
        ok(kw["thinking"] == {"type": "disabled"} and kw["output_config"]["effort"] == "low", "thinking off, low effort")
        ok(kw["max_tokens"] == persona_ai.COMPACT_MAX_TOKENS <= 3000, "small max_tokens")
        ok(kw["output_config"]["format"] == {"type": "json_schema", "schema": persona_ai.COMPACT_SCHEMA}, "structured output")
        sent = json.loads(kw["messages"][0]["content"])
        ok(sent["memos"] == ["손님은 귤을 좋아함", "손님은 귤을 아주 좋아함"] and sent["keep"] == ["내가 적은 메모"]
           and sent["impressions"][0]["name"] == "Alice" and sent["memo_target"] >= 10, "memos, keep and impressions are sent as JSON")
        ok("지시" in kw["system"] and "keep" in kw["system"], "rules: memo text is data; keep is not rewritten")
        ok(result["notes"] == ["손님은 귤을 좋아한다"] and usage == {"in": 4321, "out": 987}, "answer and token counts returned")
        messages.stop = "max_tokens"
        try:
            real.compact_memory(job)
            ok(False, "a cut-off answer must raise")
        except ValueError:
            ok(True, "a cut-off answer is refused")

        # ---- the built-in fake brain (--ai fake) never rewrites memos, it only drops exact repeats
        offline = persona_ai.Brain("fake", new_store(tmp, "data7"))
        result, usage = offline.compact_memory({"npc": "x", "notes": [{"text": "a"}, {"text": "b"}, {"text": "a"}], "keep": ["b"],
                                                "impressions": {"Alice": "x" * 450}, "before": {"count": 4, "chars": 4}})
        ok(result == {"notes": ["a"], "impressions": []} and usage == {"in": 0, "out": 0}, "offline brain: duplicates only")

        # ---- world panel ops
        bridge = cfg_bridge.CfgBridge(store)
        ok(bridge.apply({"op": "compact", "on": True}) == "" and store.settings["compact_on"] is True, "cfg op: compact on")
        ok(bridge.apply({"op": "compact_undo"}) == "" and store.settings["compact_on"] is False, "cfg op: compact_undo")

        # ---- settings page API: state, "지금 정리", "되돌리기"
        web = new_store(tmp, "data8")
        fill(web, ai=10)
        page_brain = persona_ai.Brain("fake", web)          # the offline brain; only its compaction is scripted
        scripted = FakeBrain()
        scripted.answers.append({"notes": ["손님은 귤을 좋아한다"], "impressions": []})
        page_brain.compact_memory = scripted.compact_memory
        server = app.make_server(0, web, page_brain, token="tok")
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def post(body):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("POST", "/api/save", body=json.dumps(body).encode("utf-8"),
                         headers={"Content-Type": "application/json", "X-W04-Token": "tok"})
            res = conn.getresponse()
            out = res.status, json.loads(res.read().decode("utf-8"))
            conn.close()
            return out

        try:
            code, state = post({"settings": {"compact_note_count": 40}})
            ok(code == 200 and state["settings"]["compact_note_count"] == 40 and state["compaction"]["ranges"]["compact_note_count"][0] == 5
               and state["compaction"]["now"]["count"] == 12, "page state carries thresholds, ranges and current size")
            code, state = post({"settings": {"compact_note_count": 999}})
            ok(code == 400 and "정수" in state["error"], "bad threshold refused with a Korean reason")
            code, state = post({"compact": "undo"})
            ok(code == 400 and "없습니다" in state["error"], "undo with no backup is refused")
            code, state = post({"compact": "run"})
            ok(code == 200, "'지금 정리' starts a run below the thresholds")
            deadline = time.time() + 5
            while len(web.notes) != 3 and time.time() < deadline:
                time.sleep(0.01)
            ok(len(web.notes) == 3, "manual run applied in the background")
            code, state = post({"compact": "undo"})
            ok(code == 200 and len(state["notes"]) == 12 and state["settings"]["compact_on"] is False
               and state["compaction"]["last"]["undone"], "'되돌리기' restores the memos and reports it")
            off = app.make_server(0, web, None, token="tok")
            ok(app.start_compaction(web, None, force=True) != "", "no brain (AI off): no compaction")
            off.server_close()
        finally:
            server.shutdown()
            server.server_close()
        print(f"OK: {checks} checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
