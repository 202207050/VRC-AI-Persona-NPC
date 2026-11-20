"""Offline regression scripts. Run: .venv\\Scripts\\python.exe -X utf8 tests\\test_voice_options.py"""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cfg_bridge
import persona_ai
import persona_npc_app as app
from persona_store import Store
from voice_options import MODELS, STT_VOCABULARY, clean_hints, model_options


def test_storage_and_cfg():
    with TemporaryDirectory() as tmp:
        store = Store(tmp)
        itoa = store.for_npc("itoa")
        bridge = cfg_bridge.CfgBridge(store)
        assert store.settings["brain_model"] == itoa.settings["brain_model"] == MODELS["haiku"]
        assert not bridge.apply({"npc": "itoa", "op": "model", "model": MODELS["opus"]})
        itoa.apply({"settings": {"stt_hints": "404→사공사\r\n  이토아 "}})
        saved = Store(tmp).for_npc("itoa").settings
        assert saved["brain_model"] == MODELS["opus"] and saved["stt_hints"] == "404 -> 사공사\n이토아"
        assert store.settings["stt_hints"] == "" and store.settings["brain_model"] == MODELS["haiku"]
        assert not bridge.apply({"op": "model", "model": MODELS["sonnet"]})
        assert Store(tmp).settings["brain_model"] == MODELS["sonnet"]
        assert bridge.apply({"op": "model", "npc": "itoa", "model": []})
        bridge.apply({"op": "dictate", "target": "note_add", "npc": "itoa"})
        pending = bridge.dictate_target()
        bridge.apply({"op": "dictate", "target": "note_add", "npc": "kwonjun"})
        assert bridge.dictate_apply("사공사", expected=pending)
        assert bridge.dictate_target()["npc"] == "kwonjun" and not itoa.notes and not store.notes
        for bad in ("404 ->", "A -> B -> C", "404\n404 -> 사공사", "x" * 81, "\n".join(map(str, range(33)))):
            try:
                itoa.apply({"settings": {"brain_model": MODELS["haiku"], "stt_hints": bad}})
            except ValueError:
                pass
            else:
                raise AssertionError(bad)
            assert itoa.settings == Store(tmp).for_npc("itoa").settings == saved
        itoa.apply({"settings": {"stt_hints": ""}})
        assert not Store(tmp).for_npc("itoa").settings["stt_hints"]
        path = Path(itoa.data_dir, "settings.json")
        path.write_text(json.dumps({"brain_model": "bad", "stt_hints": [], "keep_history": True}), encoding="utf-8")
        recovered = Store(tmp).for_npc("itoa").settings
        assert recovered["brain_model"] == MODELS["haiku"] and recovered["stt_hints"] == "" and recovered["keep_history"]
        # An unrelated save must not turn the CLI default into a persistent model choice.
        cli_store = Store(tmp, default_model=MODELS["sonnet"]).for_npc("itoa")
        cli_store.apply({"settings": {"stt_hints": "이토아"}})
        assert "brain_model" not in json.loads(path.read_text(encoding="utf-8"))
        assert Store(tmp).for_npc("itoa").settings["brain_model"] == MODELS["haiku"]
        assert [(o["cold_usd"], o["warm_usd"]) for o in model_options()] == [
            (0.000575, 0.000230), (0.011500, 0.004300), (0.023000, 0.008600)]


def test_model_per_reply():
    # Replace the SDK before Brain is constructed. No API key, HTTP, or .env loader.
    requests = []

    def create(**kw):
        requests.append(kw)
        data = {"lines": {"ko": "안녕!"}}
        if len(requests) == 1:
            data["search"] = "날씨"
        return SimpleNamespace(stop_reason="end_turn", usage=SimpleNamespace(input_tokens=3, output_tokens=4),
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=json.dumps(data))])

    sdk = SimpleNamespace(Anthropic=lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"anthropic": sdk}), patch.dict(os.environ, {"W04_WEB_SEARCH": "1"}):
        store = Store(tmp, str(ROOT / "personas/kwonjun.md"))
        itoa = store.for_npc("itoa")
        brain = persona_ai.Brain("haiku", itoa)
        for name, thinking in (("haiku", "disabled"), ("opus", "adaptive"), ("sonnet", "between_tools")):
            itoa.apply({"settings": {"brain_model": MODELS[name]}})
            req = brain._request_kwargs("안녕", "Alice")
            assert req["model"] == MODELS[name] and req["thinking"]["type"] == thinking
            assert req["output_config"]["effort"] == "low"
            # model name is no longer in the prompt (item 29: the NPC keeps its model secret)
            assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
        result = brain.reply("날씨", on_interim=lambda _: itoa.apply({"settings": {"brain_model": MODELS["opus"]}}))
        assert result["model"] == MODELS["sonnet"] and "error" not in result
        assert [r["model"] for r in requests] == [MODELS["sonnet"]] * 2 and "tools" in requests[1]
        assert requests[0]["system"] == requests[1]["system"]
        assert brain.reply("안녕")["model"] == MODELS["opus"]
        assert requests[-1]["thinking"] == {"type": "adaptive", "display": "omitted"}
        assert requests[-1]["messages"][0]["content"] == "날씨" and not store.history
        assert persona_ai.Brain("fake", itoa).model == "fake"
        assert app.make_brain("off", itoa) is None


def test_stt():
    audio = app.np.full(app.RATE, 0.1, dtype=app.np.float32)
    tr = app.Transcriber.__new__(app.Transcriber)  # skip Whisper/Torch load and warm-up
    tr.model, tr.device = Mock(), "cpu"
    heard = "404님, 1404 4040 user404 404abc. 사공사! 이 토아야, A+B?"
    tr.model.transcribe.return_value = {"text": heard}
    hints = clean_hints("404 -> 사공사\n사공사 -> 404\n이 -> E\n이 토아 -> 이토아\nA+B -> 별\n권준")
    got, reason, _ = app.judge(audio, 1000, tr, hints)
    assert got == "사공사님, 1404 4040 user404 404abc. 404! 이토아야, 별?" and reason.startswith("ok")
    assert tr.model.transcribe.call_args.kwargs["initial_prompt"] == ", ".join(
        (*STT_VOCABULARY, "404", "사공사", "이", "E", "이 토아", "A+B", "별"))
    assert app.judge(audio, 1000, tr)[0] == heard
    assert tr.model.transcribe.call_args.kwargs["initial_prompt"] == ", ".join(STT_VOCABULARY)
    # Transcribers without prompt support still get their existing run(audio) call.
    plain = SimpleNamespace(run=lambda audio: {"text": "404"})
    assert app.judge(audio, 1000, plain, hints)[0] == "사공사"
    tr.model.transcribe.return_value = {"text": "404", "segments": [{"no_speech_prob": 0.99, "avg_logprob": -2}]}
    assert app.judge(audio, 1000, tr, hints)[:2] == ("", "no_speech")
    tr.model.transcribe.reset_mock()
    assert not app.judge(audio * 0, 1000, tr, hints)[0] and not tr.model.transcribe.called


if __name__ == "__main__":
    test_storage_and_cfg()
    test_model_per_reply()
    test_stt()
    print("OK: store/cfg, model switching/costs, and local STT hint checks passed")
