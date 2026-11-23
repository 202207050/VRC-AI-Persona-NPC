"""Offline check of the settings page, the store and the AI's memory updates.

    .venv\\Scripts\\python.exe tests\\test_settings.py

No microphone, no Whisper, no API call: the Claude client is replaced by a fake, the server runs on a
spare port, and all files go to a temporary folder (the real persona.md / data/ are not touched).
"""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)

import persona_ai            # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store         # noqa: E402

checks = 0


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def sys_text(req):
    """The request's system prompt as one string (it is sent as cached + changing text blocks)."""
    s = req["system"]
    return s if isinstance(s, str) else "\n".join(b["text"] for b in s)


class FakeMessages:
    """Stands in for anthropic's client.messages: records the request, returns the queued JSON."""

    def __init__(self):
        self.requests, self.answers = [], []

    def create(self, **kw):
        self.requests.append(kw)
        text = json.dumps(self.answers.pop(0), ensure_ascii=False)
        return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text=text)],
                                     usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))


def answer(line, **extra):
    return dict({"emotion": "happy", "intensity": 0.5, "steps": [], "lines": {"ko": line}}, **extra)


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    h = dict(headers or {})
    if data is not None:
        h.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=h)
    res = conn.getresponse()
    text = res.read().decode("utf-8")
    conn.close()
    return res.status, text


def main():
    tmp = tempfile.mkdtemp(prefix="w04_settings_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")        # keep the real log clean
        persona = os.path.join(tmp, "persona.md")
        shutil.copy(os.path.join(ROOT, "persona.md"), persona)
        data_dir = os.path.join(tmp, "data")
        store = persona_store.Store(data_dir=data_dir, persona_path=persona)
        brain = persona_ai.Brain("fake", store)
        fake = FakeMessages()
        brain.client, brain.model = types.SimpleNamespace(messages=fake), "mock"

        # ---- store + AI, no server
        ok(store.npc_name() == "kwonjun", "name comes from persona.md")
        ok(store.speaker == persona_store.DEFAULT_SPEAKER, "no VRChat name yet -> single 'player' entry")
        log = os.path.join(tmp, "output_log_test.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.write('2026.11.19 13:14:00 Debug      -  [Behaviour] Initialized PlayerAPI "Tester One" is local\n')
        ok(app.find_local_player(log) == "Tester One", "local player name read from the VRChat log")
        store.set_speaker("Tester One")

        # t20: the reply writes no impression / memo; the memory call after it does (the app starts it)
        brain.client.with_options = lambda **_: brain.client

        def memory_call(**result):
            job = store.memory_begin("Tester One")
            fake.answers.append(dict({"changed": True, "profile": [], "impression": "", "notes": []}, **result))
            data, usage = brain.update_memory(job)
            return job, store.memory_finish(job, data, usage)

        fake.answers.append(answer("안녕!"))
        me = {"speaker": "Tester One", "source": "PC 주인 마이크"}
        r = brain.reply("안녕, 야경 예쁘다", utterance=me)
        ok(r["line"] == "안녕!" and "error" not in r, "mocked reply comes back")
        ok("impression" not in r["lines"] and set(r) >= {"emotion", "steps", "line"}, "world reply shape unchanged")
        ok(store.players["Tester One"]["talks"] == 1, "talk count")
        ok(len(store.history) == 2, "exchange stored")
        props = fake.requests[0]["output_config"]["format"]["schema"]["properties"]
        ok("impression" not in props and "memory_note" not in props, "t20: the reply schema has no memory fields")
        job, done = memory_call(impression="다정함. 야경을 좋아함.", profile=["야경을 좋아함"],
                                notes=["Tester One은 공포 영화를 못 본다"])
        call = fake.requests[-1]
        ok(call["model"] == persona_ai.MODELS["haiku"] and call["thinking"] == {"type": "disabled"}
           and call["output_config"]["effort"] == "low"
           and call["output_config"]["format"]["schema"] == persona_ai.MEMORY_SCHEMA, "memory call: Haiku, no thinking, low")
        ok(job["turns"] == [{"them": "안녕, 야경 예쁘다", "me": "안녕!"}] and json.loads(call["messages"][0]["content"])["turns"]
           == job["turns"], "the memory call reads that person's exchange")
        ok(store.players["Tester One"]["note"] == "다정함. 야경을 좋아함." and done["impression"]["after"] == "다정함. 야경을 좋아함."
           and store.players["Tester One"]["before"] == "", "AI impression stored, with the one before it")
        ok(store.players["Tester One"]["profile"] == ["야경을 좋아함"], "AI profile stored")
        ok(store.notes[0]["text"].startswith("Tester One은") and store.notes[0]["by"] == "ai", "AI memory note stored")

        fake.answers.append(answer("그치?"))
        brain.reply("응")
        second = fake.requests[-1]                                # [1] is the memory call now
        system = sys_text(second)
        blocks = second["system"]
        ok(isinstance(blocks, list) and blocks[0].get("cache_control") == {"type": "ephemeral"}
           and "Tester One" not in blocks[0]["text"] and "Tester One" in blocks[1]["text"]
           and persona_ai.content_order(0) in [b["text"] for b in blocks[2:]],
           "fixed rules+persona block is cached, the changing memory/impression comes after it")
        ok("다정함. 야경을 좋아함." in system and "공포 영화" in system and "Tester One" in system,
           "impression + memory + name are fed back into the system prompt")
        ok("- 이 사람에 대해 아는 것: 야경을 좋아함" in blocks[1]["text"], "t20: the profile sits next to the impression, uncached")
        ok(second["messages"][-1] == {"role": "user", "content": "응"} and len(second["messages"]) == 3,
           "history sent with the new line last")

        fake.answers.append(answer("x" * 10))
        brain.reply("길게")
        memory_call(impression="가" * 500, profile=["다" * 500], notes=["나" * 500])
        ok(len(store.players["Tester One"]["note"]) == persona_store.AI_IMPRESSION_CHARS, "AI impression capped")
        ok(len(store.players["Tester One"]["profile"][0]) == persona_store.PROFILE_CHARS, "AI profile line capped")
        ok(len(store.notes[-1]["text"]) == persona_store.MAX_NOTE_CHARS, "memory note capped")

        store.apply({"players": {"Tester One": {"note": "내가 직접 적음", "locked": True}}})
        fake.answers.append(answer("응"))
        brain.reply("잠금 확인")
        profile = list(store.players["Tester One"]["profile"])
        _, done = memory_call(impression="AI가 덮어쓰려 함", profile=["덮어씀"], notes=["잠금이어도 메모는 됨"])
        ok(store.players["Tester One"]["note"] == "내가 직접 적음" and store.players["Tester One"]["profile"] == profile
           and done["impression"] is None, "locked impression and profile are not overwritten")
        ok(store.notes[-1]["text"] == "잠금이어도 메모는 됨", "a locked player's memos may still be written")
        ok(store.players["Tester One"]["talks"] == 4, "talk count survives a manual edit")

        fake.answers.append({"emotion": "happy"})                 # broken answer: no line
        n = len(store.history)
        r = brain.reply("오류 확인")
        ok("error" in r and len(store.history) == n, "failed turn leaves the history untouched")

        for i in range(persona_store.MAX_NOTES + 5):
            fake.answers.append(answer("응"))
            brain.reply(f"말 {i}", utterance=me)
            memory_call(notes=[f"메모 {i}"])
        ok(len(store.notes) == persona_store.MAX_NOTES, "notes capped")
        ok(len(store.history) == 2 * persona_store.MAX_HISTORY, "history capped")
        ok(os.path.exists(os.path.join(store.data_dir, "history.json")), "t20: history is kept by default")

        # ---- server on a spare port
        server = app.make_server(0, store, brain)
        port, token = server.server_address[1], server.token
        threading.Thread(target=server.serve_forever, daemon=True).start()
        auth = {"X-W04-Token": token}
        try:
            app.replies.put({"line": "테스트", "seq": 1})
            status, text = request(port, "GET", "/poll")
            ok(status == 200 and json.loads(text) == {"line": "테스트", "seq": 1}, "/poll still works, no token")
            ok(request(port, "GET", "/ping")[1].startswith("pong"), "/ping")

            ok(request(port, "GET", "/settings")[0] == 403, "settings page without token -> 403")
            ok(request(port, "GET", "/settings?t=wrong")[0] == 403, "settings page with wrong token -> 403")
            status, text = request(port, "GET", f"/settings?t={token}")
            ok(status == 200 and "NPC 설정" in text and token in text and "__TOKEN__" not in text, "settings page with token")
            status, text = request(port, "GET", f"/settings?t={token}", headers={"Host": "evil.example:%d" % port})
            ok(status == 403, "foreign Host header -> 403 (DNS rebinding)")

            ok(request(port, "GET", "/api/state")[0] == 403, "state without token -> 403")
            ok(request(port, "GET", f"/api/state?t={token}")[0] == 403, "state needs the header, a query is not enough")
            status, text = request(port, "GET", "/api/state", headers=auth)
            state = json.loads(text)
            ok(status == 200 and state["npc_name"] == "kwonjun" and state["speaker"] == "Tester One", "state with token")
            ok(len(state["history"]) == persona_store.MAX_HISTORY and state["history"][-1]["line"] == "응", "history view")
            ok("내가 직접 적음" in state["system_preview"], "system prompt preview")
            ok(token not in text and "sk-ant" not in text, "state carries no secrets")

            body = {"persona": "# 새 성격\n- 이름: 루나\n- 조용한 아이"}
            ok(request(port, "POST", "/api/save", body)[0] == 403, "POST without token -> 403")
            ok(request(port, "POST", "/api/save", body, dict(auth, Origin="http://evil.example"))[0] == 403,
               "POST from another origin -> 403")
            ok(store.npc_name() == "kwonjun", "rejected POSTs changed nothing")
            ok(request(port, "POST", "/api/save", body, dict(auth, **{"Content-Type": "text/plain"}))[0] == 400,
               "POST must be JSON")
            status, text = request(port, "POST", "/api/save", body, dict(auth, Origin=f"http://127.0.0.1:{port}"))
            ok(status == 200 and json.loads(text)["npc_name"] == "루나", "POST save with token")
            with open(persona, encoding="utf-8") as f:
                ok("조용한 아이" in f.read(), "persona file written")
            fake.answers.append(answer("응"))
            brain.reply("성격 확인")
            ok("조용한 아이" in sys_text(fake.requests[-1]), "persona edit applies to the next reply, no restart")

            status, _ = request(port, "POST", "/api/save", {
                "settings": {"keep_history": True, "auto_update": False},
                "notes": [{"text": "손님은 커피를 좋아한다", "by": "user"}, {"text": "  "}],
                "history_delete": 0, "history_heard": store.history[0]["content"]}, auth)
            ok(status == 200 and [n["text"] for n in store.notes] == ["손님은 커피를 좋아한다"], "notes replaced")
            ok(len(store.history) == 2 * persona_store.MAX_HISTORY - 2, "one exchange deleted")
            ok(os.path.exists(os.path.join(store.data_dir, "history.json")), "keep_history on -> history.json written")
            fake.answers.append(answer("응"))
            brain.reply("자동 끔")
            ok(store.memory_claim("Tester One") is None, "auto_update off -> no memory call is started")
            _, done = memory_call(impression="무시되어야 함", notes=["무시되어야 함"])
            ok(len(store.notes) == 1 and done["notes"] == [] and done["impression"] is None,
               "auto_update off -> AI changes nothing")
            ok(request(port, "POST", "/api/save", {"history_delete": 0, "history_heard": "아닌 말"}, auth)[0] == 400,
               "stale history delete is refused")
            ok(request(port, "POST", "/api/save", {"persona": "   "}, auth)[0] == 400, "empty persona refused")
        finally:
            server.shutdown()
            server.server_close()

        # ---- restart: what survives
        again = persona_store.Store(data_dir=data_dir, persona_path=persona)
        ok(again.players["Tester One"]["note"] == "내가 직접 적음" and again.notes[0]["text"] == "손님은 커피를 좋아한다",
           "impressions and notes survive a restart")
        ok(len(again.history) == len(store.history) and again.settings["keep_history"], "history survives when kept")
        again.apply({"settings": {"keep_history": False}})
        ok(not os.path.exists(os.path.join(store.data_dir, "history.json")), "keep_history off -> history.json removed")
        ok(persona_store.Store(data_dir=data_dir, persona_path=persona).history == [], "and the next start is fresh")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_regression_checks():
    main()


if __name__ == "__main__":
    main()
