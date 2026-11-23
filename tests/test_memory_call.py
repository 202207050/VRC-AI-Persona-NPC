"""Offline checks of the memory call after a reply (t20; persona_store memory_*, persona_npc_app.start_memory).

    .venv\\Scripts\\python.exe -X utf8 tests\\test_memory_call.py

No API call: the model is a stand-in. Scheduling (one queued run per player, 10 s gap, "기억해 줘" now),
never over a hand edit, locked players, auto_update off, the undo backup and old data files.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)

import persona_ai              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402

checks = 0
ME = {"speaker": "Alice", "source": "PC 주인 마이크"}


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def new_store(tmp, name):
    persona = os.path.join(tmp, name + "_persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    return persona_store.Store(data_dir=os.path.join(tmp, name), persona_path=persona)


def talk(st, text="안녕", line="반가워", who="Alice"):
    st.after_reply(who, text, json.dumps({"lines": {"ko": line}}, ensure_ascii=False), utterance=dict(ME, speaker=who))


def answer(**kw):
    return dict({"changed": True, "profile": [], "impression": "", "notes": []}, **kw)


def schedule_checks(tmp):
    st = new_store(tmp, "sched")
    ok(st.memory_claim("Alice", now=100.0) == 0.0, "first call: no wait")
    ok(st.memory_claim("Alice", now=100.5) is None, "a second turn while it runs starts nothing ...")
    ok(st.memory_claim("Alice", now=100.6) is None and st._memory_busy["Alice"] == "again", "... and only ONE run is queued")
    talk(st)
    ok(st.memory_begin("Alice", now=101.0) is not None, "the job reads the exchange")
    ok(st.memory_release("Alice", now=102.0) == 9.0, "the queued run waits for the 10 s gap")
    ok(st.memory_release("Alice", now=112.0) is None and "Alice" not in st._memory_busy, "nothing queued: claim dropped")
    ok(st.memory_claim("Alice", now=103.0) == 8.0, "a new turn within 10 s waits the rest of the gap")
    ok(st.memory_claim("Bob", now=103.0) == 0.0, "each player has their own gap")
    ok(st.memory_claim("Alice", force=True, now=103.5) is None and st._memory_busy["Alice"] == "force",
       "'기억해 줘' while busy: the queued run is forced")
    ok(st.memory_release("Alice", now=104.0) == 0.0, "a forced run skips the gap")
    st.memory_release("Alice", now=104.0)
    ok(st.memory_wait("Alice", force=True, now=101.5) == 0.0, "force = no wait")
    st.apply({"settings": {"auto_update": False}})
    ok(st.memory_claim("Carol", now=200.0) is None, "auto_update off: no memory call")
    st.apply({"settings": {"auto_update": True}})
    ok(st.memory_begin("Nobody") is None, "no exchange with that player: nothing to read")
    for text in ("이거 기억해 줘", "잊지 마", "외워 둬", "remember this", "기억해둬"):
        ok(persona_store.asks_to_remember(text), f"remember request: {text}")
    for text in ("기억나?", "저번 일 기억 안 나", "안녕"):
        ok(not persona_store.asks_to_remember(text), f"not a remember request: {text}")


def finish_checks(tmp):
    st = new_store(tmp, "finish")
    for i in range(8):
        talk(st, f"말 {i}", f"답 {i}")
    talk(st, "옆 사람 말", "응", who="Bob")
    job = st.memory_begin("Alice")
    ok(len(job["turns"]) == persona_store.MEMORY_TURNS and job["turns"][-1] == {"them": "말 7", "me": "답 7"}
       and all("옆 사람" not in t["them"] for t in job["turns"]), "the last 6 turns with that player only")
    ok(set(job) == {"npc", "name", "label", "talks", "turns", "profile", "impression", "notes"}, "job fields")
    st.set_impression("Alice", "내가 손으로 적음")                 # edited on the settings page meanwhile
    done = st.memory_finish(job, answer(impression="AI 인상", profile=["귤을 좋아함"], notes=["Alice는 귤을 좋아한다"]),
                            {"in": 1000, "out": 100})
    p = st.players["Alice"]
    ok(p["note"] == "내가 손으로 적음" and done["impression"] is None, "a hand edit during the call is not overwritten")
    ok(p["profile"] == ["귤을 좋아함"] and done["notes"] == ["Alice는 귤을 좋아한다"], "profile and memo still stored")
    ok(done["usage"] == {"in": 1000, "out": 100} and done["cost_usd"] > 0, "usage and cost are reported")
    ok(st.memory_changes == 1 and st.memory_changed_at == done["at"], "the session change count")

    job = st.memory_begin("Alice")
    done = st.memory_finish(job, answer(changed=False, impression="무시", notes=["무시"]))
    ok(done["ok"] and done["impression"] is None and done["notes"] == [] and st.memory_changes == 1, "changed=false: nothing")
    done = st.memory_finish(job, None, None, "TimeoutError")
    ok(not done["ok"] and done["error"] == "TimeoutError" and st.memory_changes == 1, "a failed call changes nothing")
    done = st.memory_finish(job, answer(impression="새 인상", profile=[], notes=["Alice는 귤을 좋아한다", "없음", "-"]))
    ok(st.players["Alice"]["note"] == "새 인상" and st.players["Alice"]["before"] == "내가 손으로 적음"
       and st.players["Alice"]["profile"] == ["귤을 좋아함"] and done["notes"] == [],
       "impression with the one before it; an empty profile never wipes it; repeats and none-markers are skipped")
    job = st.memory_begin("Alice")
    done = st.memory_finish(job, answer(profile=["a", "b", "b", "c", "d", "e", "없음"], notes=["1", "2", "3"]))
    ok(st.players["Alice"]["profile"] == ["a", "b", "c", "d"] and done["notes"] == ["1", "2"],
       "profile: 4 lines, no repeats; at most 2 new memos")

    st.set_locked("Alice", True)
    job = st.memory_begin("Alice")
    done = st.memory_finish(job, answer(impression="잠금 무시", profile=["잠금 무시"], notes=["잠겨도 메모"]))
    ok(st.players["Alice"]["note"] == "새 인상" and st.players["Alice"]["profile"] == ["a", "b", "c", "d"]
       and done["notes"] == ["잠겨도 메모"], "locked: impression and profile kept, memos still written")
    st.set_locked("Alice", False)

    # undo: the newest backup is the memory call; it is taken back without switching compaction off
    job = st.memory_begin("Alice")
    st.memory_finish(job, answer(impression="되돌릴 인상", profile=["되돌릴 줄"], notes=["되돌릴 메모"]))
    before = dict(st.settings)
    ok(st.compaction_undo() == "", "undo works on a memory call")
    texts = [n["text"] for n in st.notes]
    ok(st.players["Alice"]["note"] == "새 인상" and st.players["Alice"]["profile"] == ["a", "b", "c", "d"]
       and "되돌릴 메모" not in texts and "잠겨도 메모" in texts, "impression, profile and the memo are put back")
    ok(st.settings["compact_on"] == before["compact_on"], "undoing a memory call leaves auto compaction alone")

    st.apply({"settings": {"auto_update": False}})
    done = st.memory_finish(st.memory_begin("Alice"), answer(impression="꺼짐", notes=["꺼짐"]))
    ok(done["impression"] is None and done["notes"] == [], "auto_update switched off meanwhile: nothing stored")


def old_file_checks(tmp):
    folder = os.path.join(tmp, "old", "npcs", "kwonjun")
    os.makedirs(folder)
    pat = persona_store.AFFECTION_PHRASES["pat"]
    with open(os.path.join(folder, "impressions.json"), "w", encoding="utf-8") as f:
        json.dump({"players": {"앨리스": {"note": "다정함. " + pat, "locked": False, "updated": "2026-11-19 10:00",
                                          "talks": 3}}}, f, ensure_ascii=False)
    st = new_store(tmp, "old")
    p = st.players["앨리스"]
    ok(p["profile"] == [] and p["note"] == "다정함." and p["touch_recorded"] == ["pat"],
       "old data: an empty profile, the fixed phrase moved out of the impression")
    ok(st.settings["keep_history"] is True, "keep_history is on by default")


def app_checks(tmp):
    """persona_npc_app.start_memory: runs after the reply, never twice at once, then compaction."""
    st = new_store(tmp, "app")
    gate, calls = threading.Event(), []

    class Brain:
        def update_memory(self, job):
            calls.append(job["turns"][-1]["them"])
            gate.wait(5)
            return answer(notes=[f"메모 {len(calls)}"]), {"in": 10, "out": 5}

        def compact_memory(self, job):
            return {"notes": [n["text"] for n in job["notes"]], "impressions": []}, {"in": 0, "out": 0}

    talk(st, "첫 말")
    app.start_memory(st, Brain(), "Alice")
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    talk(st, "둘째 말")
    app.start_memory(st, Brain(), "Alice", force=True)          # queued (forced: no 10 s gap)
    app.start_memory(st, Brain(), "Alice", force=True)          # still one queued run
    gate.set()
    deadline = time.time() + 5
    while (len(calls) < 2 or "Alice" in st._memory_busy) and time.time() < deadline:
        time.sleep(0.01)
    ok(calls == ["첫 말", "둘째 말"], f"one run, then one queued run that reads the newest turn ({calls})")
    ok([n["text"] for n in st.notes][-2:] == ["메모 1", "메모 2"] and st.memory_changes == 2, "both runs stored")
    with open(app.LOG_PATH, encoding="utf-8") as f:
        logged = f.read()
    ok(logged.count("\tMEMORY_CALL\tnpc=kwonjun\twho=Alice\tok=1\tchanged=1") == 2, "MEMORY_CALL logged per run")


def main():
    tmp = tempfile.mkdtemp(prefix="w04_memory_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")
        schedule_checks(tmp)
        finish_checks(tmp)
        old_file_checks(tmp)
        app_checks(tmp)
        ok(set(persona_ai.MEMORY_SCHEMA["required"]) == set(persona_ai.MEMORY_SCHEMA["properties"])
           == {"changed", "profile", "impression", "notes"}, "memory schema: every field required")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_memory_call_checks():
    main()


if __name__ == "__main__":
    main()
