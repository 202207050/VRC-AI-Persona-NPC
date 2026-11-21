"""Offline checks for the current-world context, address rules and timing diagnostics."""
import json
import os
import re
import tempfile
import threading
from urllib.request import urlopen

from test_gaze_listen import LISTEN_ON, LOCAL, T, add, join, room, speech, start_watcher, tsv, wait_for
from test_websearch import FakeMessages, NS, answer, response, text, wants

import local_commands
import persona_ai
import persona_npc_app as app
import numpy as np


def test_tv_state():
    state = {}

    def event(detail):
        line = f"{T}[W04AI] {detail}"
        app.update_tv_state(state, app.EVENT_RE.search(line)[1], line)

    event("TV_STATE inches=160 min=120 max=300 backlight=42")
    assert app.tv_context("TV 화면 몇 인치야?", state) == "[TV 현재: 화면 160인치(최소 120, 최대 300), 백라이트 42%]"
    assert state["at"] > 0
    assert app.tv_context("안녕", state) is None
    event("TV_SIZE_DONE npc=itoa inches=200 fits=False")
    event("TV_BACKLIGHT npc=itoa request=60 percent=-1")
    assert state["inches"] == 160 and state["backlight"] == 42
    state.clear()
    event("TV_SIZE_DONE npc=itoa inches=180 fits=True")
    assert app.tv_context("화면 크기", state) == "[TV 현재: 화면 180인치]"
    event("TV_BACKLIGHT npc=itoa request=0 percent=0")
    assert app.tv_context("밝기", state) == "[TV 현재: 화면 180인치, 백라이트 0%]"
    assert app.tv_context("밝기", {}) is None
    assert app.tv_context("TV", {"min": 120}) == "[TV 현재: 최소 120인치]"


def test_rules():
    for heard in ("토아 이리 와", "토아야!", "토아한테, 물어봐", "이토하 앉아", "ITOAR, 이리 와"):
        assert app.npc_named(heard, "itoa", "ITOAR"), heard
    for heard in ("토아리 먹자", "도토아", "이토록", "권준아"):
        assert not app.npc_named(heard, "itoa", "ITOAR"), heard
    assert app.npc_named("권 쥰아", "kwonjun", "kwonjun")
    assert local_commands.world_command("나이트 모드 20% 낮춰 줘") is None
    assert local_commands.world_command("나이트 모드 20%만큼 올려 줘") is None
    assert local_commands.world_command("나이트 모드 20%") == {"do": "night_mode", "action": "20"}
    assert "여러분" in persona_ai.ADDRESS_ORDER
    for raw, expected in [
        (["look_at", "gesture", "tv_reset", "tv_closer_max"], ["look_at", "tv_reset", "tv_closer_max"]),
        (["look_at", "gesture", "look_at", "tv_left"], ["look_at", "gesture", "tv_left"]),
        (["look_at", "gesture", "look_at", "wait"], ["look_at", "gesture", "look_at"]),
        (["bad", "look_at", "bad", "gesture"], ["look_at"]),
        (["look_at", "gesture", "look_at"] + ["bad"] * 4 + ["tv_left", "tv_reset"], ["look_at", "gesture", "tv_left"]),
    ]:
        assert [row["do"] for row in persona_ai.clamp_steps([{"do": step} for step in raw])] == expected


def test_watch_and_calls():
    fake = FakeMessages()

    def make_brain(store):
        brain = persona_ai.Brain("fake", store)
        brain.client, brain.model = NS(messages=fake), "mock"
        return brain

    with tempfile.TemporaryDirectory() as folder:
        old_log = app.LOG_PATH
        app.LOG_PATH = os.path.join(folder, "voice_log.tsv")
        try:
            path, store, _, brain, tr, mic, stop, watcher = start_watcher(folder, "tv_state", "gaze", make_brain)
            try:
                assert "화면·녹화·사진·방송이 어떻게 보이는지 볼 수 없다" in brain.system_prompt()
                assert "화면 보기, 브라우저 조작과 월드 밖 행동은 못 한다." in brain.system_prompt()
                add(path, T + "[W04AI] TV_STATE inches=160 min=120 max=300 backlight=42\n" + LISTEN_ON)
                wait_for(lambda: mic.tap is not None, "TV snapshot read and listener open")

                def speak(heard):
                    tr.text = heard
                    mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))

                fake.queue.append(response([text(answer("지금은 160인치야."))]))
                spoke = "권준아 화면 몇 인치야?"
                with app.make_server(0, store, token="test-token") as server:
                    serving = threading.Thread(target=server.serve_forever, daemon=True)
                    serving.start()
                    try:
                        speak(spoke)
                        wait_for(lambda: not app.replies.empty(), "reply queued")
                        for _ in range(3):
                            with urlopen(f"http://127.0.0.1:{server.server_port}/poll", timeout=5) as result:
                                reply = json.load(result)
                            if reply.get("line"):
                                break
                        wait_for(lambda: tsv("POLL_LINES"), "poll delivery logged")
                    finally:
                        server.shutdown()
                        serving.join(3)
                assert reply["line"] == "지금은 160인치야." and "vad_end_at" not in reply
                request = fake.requests[-1]
                assert "[TV 현재: 화면 160인치(최소 120, 최대 300), 백라이트 42%]" in request["messages"][-1]["content"]
                assert "for_me" not in request["output_config"]["format"]["schema"]["required"]
                assert [row["content"] for row in store.history if row["role"] == "user"] == [spoke]
                assert "[TV 현재:" not in str(store.history)
                seq = reply["seq"]
                wait_for(lambda: tsv("AI_CALLS"), "diagnostics emitted")
                timing = dict(field.split("=") for field in tsv("TIMING")[-1].split("\t"))
                stages = ["vad_end_ms", "queue_ms", "stt_start_ms", "stt_end_ms",
                          "ai_start_ms", "ai_end_ms", "reply_queued_ms"]
                times = [int(timing[key]) for key in stages]
                assert timing["seq"] == str(seq) and times[0] == 0 and times == sorted(times)
                poll = dict(field.split("=") for field in tsv("POLL_LINES")[-1].split("\t"))
                assert poll["seq"] == str(seq) and int(poll["since_vad_ms"]) >= times[-1]
                assert re.fullmatch(rf"seq={seq} first:\d+:3:0:[0-9a-f]{{8}}:[0-9a-f]{{8}}:disabled:low",
                                    tsv("AI_CALLS")[-1])

                speak("토아한테, 이리 와")
                wait_for(lambda: tsv("AI_SKIP"), "other NPC named, skip")
                assert len(fake.requests) == 1 and "named=0\trecent=1" in tsv("AI_SKIP")[-1]
                add(path, join(LOCAL) + LISTEN_ON)
                wait_for(lambda: len(tsv("LISTEN_ON")) == 2, "same world joined again")
                fake.queue.append(response([text(answer("지금 값은 몰라."))]))
                speak("권준아 TV 몇 인치야?")
                assert app.replies.get(timeout=5)["line"] == "지금 값은 몰라."
                assert "[TV 현재:" not in fake.requests[-1]["messages"][-1]["content"]
            finally:
                stop.set()
                watcher.join(3)

            # Same system, model and tool names: changing only the schema changes only its fingerprint.
            brain.search_on = lambda: True
            fake.queue.extend([wants("서울 날씨"), response([text(answer("서울은 맑아."))])])
            result = brain.reply("서울 날씨 찾아봐")
            first, search = result["calls"]
            assert [first["kind"], search["kind"]] == ["first", "search"]
            assert all(re.fullmatch(r"[0-9a-f]{8}", call[key])
                       for call in (first, search) for key in ("fp_system", "fp_schema"))
            assert first["fp_system"] != search["fp_system"] and first["fp_schema"] != search["fp_schema"]
            fake.queue.append(response([text(answer("안녕!"))]))
            same = brain.reply("안녕")["calls"][0]
            assert first["fp_system"] == same["fp_system"] and first["fp_schema"] == same["fp_schema"]
            fake.queue.append(response([text(answer("응!", for_me=True))]))
            changed = brain.reply("잘 지내?", check_address=True)["calls"][0]
            assert changed["fp_system"] == first["fp_system"] and changed["fp_schema"] != first["fp_schema"]
        finally:
            app.LOG_PATH = old_log


if __name__ == "__main__":
    for check in (test_tv_state, test_rules, test_watch_and_calls):
        check()
        print("PASS", check.__name__)
