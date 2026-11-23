"""Offline TV size conversion, schema, and voice reply delivery checks."""
import json
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from test_gaze_listen import LISTEN_ON, LOCAL, T, add, join, room, speech, start_watcher, tsv, wait_for
from test_websearch import FakeMessages, NS, answer, response, text

import persona_ai
import persona_npc_app as app


STATE = {"inches": 160, "min": 120, "max": 300, "backlight": 42}


def size(**fields):
    return {"do": "tv_size_set", **fields}


def converted(request, state=STATE):
    reply = persona_ai._clamp_reply({"steps": [request], "lines": {"ko": "맞출게."}})
    return app.resolve_tv_sizes(reply, state)["steps"]


def test_boundaries_and_clipping():
    for request, inches in (
        (size(percent=0), 120), (size(percent=50), 210), (size(percent=80), 260),
        (size(percent=100), 300), (size(percent_delta=0), 160),
        (size(percent_delta=20), 190), (size(percent_delta=-20), 130),
        (size(percent_delta=-100), 120), (size(percent_delta=300), 300),
        (size(width_m=3), 140), (size(width_m=2.5), 120), (size(height_m=2), 160),
        (size(height_m=5), 300), (size(inches=183.5), 183.5),
    ):
        assert converted(request) == [size(inches=inches, target="none", gesture="none")], request
    assert converted(size(percent=50), dict(STATE, min=120, max=130))[0]["inches"] == 120
    assert converted(size(percent=50), dict(STATE, min=130, max=140))[0]["inches"] == 140
    assert converted(size(percent_delta=-42), dict(STATE, inches=250))[0]["inches"] == 140
    assert converted(size(percent=50), dict(STATE, max=120))[0]["inches"] == 120
    for request in (size(), size(inches=180, width_m=3), size(width_m=3, height_m=2),
                    size(percent=101), size(percent=-1), size(percent_delta=-101), size(percent_delta=301)):
        assert converted(request) == [], request
    for key in ("inches", "percent", "percent_delta", "width_m", "height_m"):
        for value in (True, "3", None, float("nan"), float("inf"), -float("inf")):
            assert converted(size(**{key: value})) == [], (key, value)
    for key in ("width_m", "height_m"):
        for value in (0, -1):
            assert converted(size(**{key: value})) == [], (key, value)


def test_missing_state_and_dialogue():
    missing = "지금 TV 크기 정보를 못 받아서, 인치로 말해 줘."
    pause = {"do": "tv_pause", "target": "none", "gesture": "none"}
    for request in (size(percent=50), size(percent_delta=20)):
        alone = persona_ai._clamp_reply({"steps": [request, {"do": "look_at", "target": "tv"}],
                                         "lines": {"ko": "맞출게."}})
        # t19: only the size step is left out; another command still runs and the line says why
        reply = persona_ai._clamp_reply({"steps": [request, {"do": "tv_pause"}],
                                         "lines": {"ko": "맞출게.", "en": "Sure.", "ja": "うん。"}}, global_mode=True)
        for state in (None, {}, {"backlight": 42}, {"min": 120, "max": 300},
                      {"inches": 160, "min": 120}, dict(STATE, max=100)):
            result = app.resolve_tv_sizes(alone, state, global_mode=True)
            assert result["steps"] == [] and result["line"] == missing
            assert set(result["lines"]) == {"ko", "en", "ja"}
            assert all(result["lines"].values())
            result = app.resolve_tv_sizes(reply, state, global_mode=True)
            assert result["steps"] == [pause]
            assert result["lines"] == {"ko": "맞출게. " + missing, "en": "Sure. " + app.TV_SIZE_STATE_MISSING[1],
                                       "ja": "うん。 " + app.TV_SIZE_STATE_MISSING[2]}
            assert result["line"] == result["lines"]["ko"]
        assert set(app.resolve_tv_sizes(alone, {})["lines"]) == {"ko"}
        named = persona_ai._clamp_reply({"steps": [request, {"do": "mirror", "action": "lq"}],
                                         "lines": {"ko": "50%로 맞출게."}})
        result = app.resolve_tv_sizes(named, {})
        assert result["line"] == missing and [s["do"] for s in result["steps"]] == ["mirror"]
    assert converted(size(width_m=2.5), {})[0]["inches"] == 110
    assert converted(size(height_m=2), None)[0]["inches"] == 160
    assert converted(size(width_m=0.001), {})[0]["inches"] == 10
    assert converted(size(height_m=1e308), {})[0]["inches"] == 1_000_000
    assert converted(size(inches=180), {})[0]["inches"] == 180

    reply = persona_ai._clamp_reply({"steps": [size(width_m=3), {"do": "tv_backlight", "percent": 35}],
                                     "lines": {"ko": "가로 3m로 맞출게."}})
    result = app.resolve_tv_sizes(reply, {})
    assert result["line"] == reply["line"] and result["lines"] == reply["lines"]
    assert result["steps"][0]["inches"] == 140 and "width_m" not in result["steps"][0]
    assert reply["steps"][0]["width_m"] == 3
    assert result["steps"][1] == reply["steps"][1] and result["steps"][1]["percent"] == 35


def test_schema_and_tv_context():
    schema = persona_ai.reply_schema()
    properties = schema["properties"]["steps"]["items"]["properties"]
    assert properties["amount"]["type"] == "number"
    assert properties["unit"]["enum"] == ["percent", "percent_delta", "inches", "width_m", "height_m"]
    for key in ("inches", "percent", "percent_delta", "width_m", "height_m"):
        assert key not in properties
    for example in ("TV 50%로", "화면 크기 80퍼센트로", "화면 30% 더 크게", "폭 2.5m로",
                    "화면 가로 300센티로", "세로 2m로", "tv_size_set이 아니라 tv_backlight", "밝기 15%로"):
        assert example in persona_ai.RULES
    expected = "[TV 현재: 화면 160인치(최소 120, 최대 300), 백라이트 42%]"
    for heard in ("TV 50%로", "가로 3미터로", "폭 2.5m로", "세로 2m로", "화면 가로 300센티로"):
        assert app.tv_context(heard, STATE) == expected
    assert app.tv_context("가로로 걸어 줘", STATE) is None


def test_voice_delivery():
    fake = FakeMessages()

    def make_brain(store):
        brain = persona_ai.Brain("fake", store)
        brain.client, brain.model = NS(messages=fake), "mock"
        return brain

    with TemporaryDirectory() as folder, patch.object(app, "LOG_PATH", os.path.join(folder, "voice_log.tsv")):
        path, store, _, _, tr, mic, stop, watcher = start_watcher(folder, "tvsize", "gaze", make_brain)
        try:
            add(path, T + "[W04AI] TV_STATE inches=160 min=120 max=300 backlight=42\n" + LISTEN_ON)
            wait_for(lambda: mic.tap is not None, "TV snapshot read and listener open")
            fake.queue.append(response([text(answer("50%로 맞출게.", steps=[size(percent=50)]))]))
            tr.text = "권준아 TV 50%로"
            mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
            reply = app.replies.get(timeout=5)
            assert reply["line"] == "50%로 맞출게."
            assert reply["steps"] == [size(inches=210, target="none", gesture="none")]
            prompt = fake.requests[-1]["messages"][-1]["content"]
            assert "[TV 현재: 화면 160인치(최소 120, 최대 300), 백라이트 42%]" in prompt

            add(path, join(LOCAL) + LISTEN_ON)
            wait_for(lambda: len(tsv("LISTEN_ON")) == 2, "same world joined again")
            fake.queue.append(response([text(answer("50%로 맞출게.", steps=[size(percent=50)]))]))
            mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
            reply = app.replies.get(timeout=5)
            assert reply["line"] == "지금 TV 크기 정보를 못 받아서, 인치로 말해 줘."
            assert reply["steps"] == []
            saved = json.loads(store.history[-1]["content"])
            assert saved["steps"] == [] and saved["lines"] == reply["lines"]
        finally:
            stop.set()
            watcher.join(3)


def test_relative_change():
    """t128 re-score (EV-029/039): "20% 낮춰" is a change from the current value, even when the model sets 20%."""
    assert persona_ai.relative_percent("NPC 백라이트 조절 20% 낮춰줘") == -20
    assert persona_ai.relative_percent("나이트 모드 20%만큼 올려 줘") == 20
    assert persona_ai.relative_percent("TV 30퍼센트 키워") == 30
    assert persona_ai.relative_percent("밝기 15%로 바꿔") is None
    heard = "백라이트 20% 낮춰"
    for model in ({"do": "tv_backlight", "percent": 20}, {"do": "tv_backlight", "action": "dimmer"},
                  {"do": "tv_backlight", "percent_delta": -20}):
        reply = persona_ai._clamp_reply({"steps": [model], "lines": {"ko": "낮출게."}}, heard=heard)
        out = app.resolve_tv_sizes(reply, STATE)
        assert out["steps"] == [{"do": "tv_backlight", "target": "none", "gesture": "none", "percent": 22}]
        assert out["line"] == "낮출게."
        out = app.resolve_tv_sizes(reply, {})
        assert out["steps"] == [{"do": "tv_backlight", "target": "none", "gesture": "none", "action": "dimmer"}]
        assert out["line"] == "낮출게. " + persona_ai.STEP_NOTES["backlight_step"][0]
    full = app.resolve_tv_sizes(persona_ai._clamp_reply(
        {"steps": [{"do": "tv_backlight", "percent_delta": 90}], "lines": {"ko": "올릴게."}}), STATE)
    assert full["steps"][0]["percent"] == 100
    # Size: same merged path as the existing percent_delta conversion (160 + 20% of 160 = 192).
    reply = persona_ai._clamp_reply({"steps": [size(percent=20)], "lines": {"ko": "키울게."}}, heard="TV 20% 키워")
    assert app.resolve_tv_sizes(reply, STATE)["steps"] == converted(size(percent_delta=20))
    # The model's own sum from the [TV 현재] row is kept; only N sent back as a target is corrected.
    kept = persona_ai._clamp_reply({"steps": [{"do": "tv_backlight", "percent": 22}],
                                    "lines": {"ko": "42%에서 22%로 낮출게."}}, heard=heard)
    assert app.resolve_tv_sizes(kept, {})["steps"][0]["percent"] == 22
    # A line naming the wrong % is replaced by what was actually done.
    wrong = persona_ai._clamp_reply({"steps": [{"do": "tv_backlight", "percent": 20}],
                                     "lines": {"ko": "응, 20% 낮출게!"}}, heard=heard)
    assert app.resolve_tv_sizes(wrong, STATE)["line"] == "백라이트를 42%에서 22%로 맞출게."
    assert app.resolve_tv_sizes(wrong, {})["line"] == persona_ai.STEP_NOTES["backlight_step"][0]
    # The evals hand the state over the way tv_context wrote it.
    row = app.tv_context("TV 밝기", STATE)
    assert app.tv_state_from_context("x\n" + row) == STATE
    night = persona_ai._clamp_reply({"steps": [{"do": "night_mode", "action": "20"}],
                                     "lines": {"ko": "20%로 맞출게!"}}, heard="나이트 모드 20%만큼 올려 줘")
    assert night["steps"] == [] and night["line"] == persona_ai.STEP_NOTES["night_relative"][0]


if __name__ == "__main__":
    for check in (test_boundaries_and_clipping, test_missing_state_and_dialogue,
                  test_schema_and_tv_context, test_relative_change, test_voice_delivery):
        check()
        print("PASS", check.__name__)
