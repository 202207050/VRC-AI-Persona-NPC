r"""Offline checks. Run with .venv\Scripts\python.exe -X utf8 tests/test_smalltalk.py."""
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import persona_ai
import persona_store
import smalltalk


def test_display_names():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        personas = root / "personas"
        personas.mkdir()
        for npc, old in (("kwonjun", "권준"), ("itoa", "이토아")):
            (personas / f"{npc}.md").write_text(f"# {old} 성격 설정\n- 이름: {old}\n- 별빛을 좋아해\n", encoding="utf-8")
        with patch.object(persona_store, "HERE", str(root)), patch.dict("os.environ", {}, clear=True):
            store = persona_store.Store(str(root / "data"))
            for npc, name in (("kwonjun", "kwonjun"), ("itoa", "ITOAR")):
                st = store.for_npc(npc)
                assert st.npc_name() == st.state()["npc_name"] == name
                assert (personas / f"{npc}.md").read_text(encoding="utf-8") == f"# {name} 성격 설정\n- 이름: {name}\n- 별빛을 좋아해\n"
                assert not st.set_npc_name("이토아 친구")
                reloaded = persona_store.Store(str(root / "data"), npc=npc)
                assert reloaded.npc_name() == "이토아 친구"
                assert persona_store.Store(str(root / "data"), str(root / "missing.md"), npc=npc).npc_name() == name
            # Read-only exports can use the same display migration without touching a source persona.
            legacy = root / "legacy.md"
            legacy.write_text("- 이름: 권준\n", encoding="utf-8")
            export = persona_store.Store(str(root / "data"), str(legacy), migrate_persona=False)
            assert export.npc_name() == "kwonjun" and legacy.read_text(encoding="utf-8") == "- 이름: 권준\n"


def test_parse_request():
    prefix = "2026.11.19 13:44:37 Debug      -  [W04AI] "
    for pair in ("kwonjun,itoa", "itoa,kwonjun"):
        assert smalltalk.parse_request(prefix + f"NPC_CHAT_REQ npcs={pair} watcher=손님 이름 404\r\n") == (
            ("kwonjun", "itoa"), "손님 이름 404")
    for bad in ("NPC_CHAT_REQX npcs=kwonjun,itoa watcher=A", "NPC_CHAT_REQ npcs=kwonjun,kwonjun watcher=A",
                "NPC_CHAT_REQ npcs=kwonjun,ITOAR watcher=A", "NPC_CHAT_REQ npcs=kwonjun watcher=A",
                "NPC_CHAT_REQ npcs=kwonjun,itoa,itoa watcher=A", "NPC_CHAT_REQ npcs=kwonjun,itoa",
                "NPC_CHAT_REQ npcs=kwonjun,itoa watcher=  ", "NPC_CHAT_REQ npcs=kwonjun,itoa watcher=A\nextra"):
        assert smalltalk.parse_request(prefix + bad) is None, bad
    assert smalltalk.parse_request("NPC_CHAT_REQ npcs=kwonjun,itoa watcher=A") is None


def test_picker():
    chat = smalltalk.SmallTalk(None, random.Random(42))
    originals = json.loads(smalltalk.SCRIPT.read_text(encoding="utf-8"))["kwonjun,itoa"]
    assert len(chat.scripts) == len(originals) >= 40
    assert all([{k: v for k, v in item.items() if k != "lines"} for item in raw] == smalltalk.clamp_dialog(raw)
               for raw in originals)
    assert all(all({"en", "ja"} <= item["lines"].keys() for item in dialog) for dialog in chat.scripts)
    assert all("lines" not in item for item in chat.scripted())
    assert all(item["lines"]["ko"] == item["line"] for item in chat.scripted(global_mode=True))
    assert len({tuple(item["line"] for item in dialog) for dialog in originals}) == len(originals)
    recent = []
    for _ in range(100):
        dialog = chat.scripted()
        lines = tuple(item["line"] for item in dialog)
        assert lines not in recent[-10:]
        recent.append(lines)
        dialog[0]["line"] = "mutated reply"


def test_daily_cap():
    with tempfile.TemporaryDirectory() as folder:
        store = persona_store.Store(folder, str(ROOT / "personas/kwonjun.md"))
        other = store.for_npc("itoa")
        assert store.smalltalk_settings() == {"ai": True, "ratio": 0.2, "daily_cap": 30}
        with patch.object(persona_store.time, "strftime", return_value="2026-11-19") as date:
            assert not store.reserve_smalltalk_ai(0.2)
            assert all(other.reserve_smalltalk_ai(0.19) for _ in range(30))
            assert not store.reserve_smalltalk_ai(0)
            again = persona_store.Store(folder, str(ROOT / "personas/kwonjun.md"))
            assert not again.reserve_smalltalk_ai(0)
            date.return_value = "2026-11-20"
            other.set_smalltalk_settings({"ai": False, "ratio": 1, "daily_cap": 1.0})
            assert store.smalltalk_settings() == {"ai": False, "ratio": 1, "daily_cap": 1}
            assert not store.reserve_smalltalk_ai(0)
            store.set_smalltalk_settings({"ai": True})
            assert again.reserve_smalltalk_ai(0) and not again.reserve_smalltalk_ai(0)
            for patch_values in ({"ratio": 1.1}, {"ratio": float("nan")}, {"daily_cap": 0.5}, {"ai": "off"}):
                try:
                    store.set_smalltalk_settings(patch_values)
                except ValueError:
                    pass
                else:
                    raise AssertionError(patch_values)
            assert store.smalltalk_settings() == {"ai": True, "ratio": 1, "daily_cap": 1}


def test_ai_path():
    with tempfile.TemporaryDirectory() as folder:
        store = persona_store.Store(folder, str(ROOT / "personas/kwonjun.md"))
        itoa = store.for_npc("itoa")
        store.add_note("밤 영화 메모")
        itoa.add_note("귤 수첩 메모")
        store.set_speaker("말하지 않을 사람")
        store.set_smalltalk_settings({"ratio": 1, "daily_cap": 2})
        brain = persona_ai.Brain("fake", store)
        client = Mock()
        client.with_options.return_value = client
        brain.client = client
        brain.model = "fake"
        fake_dialog = [{"npc": "kwonjun", "line": "  귤  1000개네 ", "emotion": "invalid", "intensity": 4, "gesture": "bad"},
                       {"npc": "itoa", "line": "귤" * 120, "emotion": "happy", "intensity": -2, "gesture": "nod"}]
        client.messages.create.return_value = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(
            type="text", text=json.dumps({"dialog": fake_dialog}, ensure_ascii=False))])
        chat = smalltalk.SmallTalk(store, random.Random(42))
        idle = lambda: True
        payload, source = chat.reply(brain, idle)
        assert source == "ai" and payload["npc"] == "kwonjun" and payload["smalltalk"] is True
        assert payload["dialog"][0] == {"npc": "kwonjun", "line": "귤 1,000개네", "emotion": "neutral",
                                       "intensity": 1.0, "gesture": "none"}
        assert payload["dialog"][1]["line"] == "귤" * 80 + "…"
        assert payload["dialog"][1]["intensity"] == 0
        client.with_options.assert_called_once_with(max_retries=0)
        client.messages.create.assert_called_once()
        request = client.messages.create.call_args.kwargs
        assert request["model"] == persona_ai.MODELS["haiku"] and "tools" not in request
        profiles = json.loads(request["messages"][0]["content"])
        assert [p["name"] for p in profiles] == ["kwonjun", "ITOAR"]
        assert [p["notes"] for p in profiles] == [["밤 영화 메모"], ["귤 수첩 메모"]]
        assert "검은 니트" in profiles[0]["persona"] and "후드" in profiles[1]["persona"]
        assert "말하지 않을 사람" not in request["messages"][0]["content"]
        assert not store.history and not itoa.history
        fake_dialog[1]["npc"] = "kwonjun"
        client.messages.create.return_value.content[0].text = json.dumps({"dialog": fake_dialog})
        assert chat.reply(brain, idle)[1] == "fallback"
        assert chat.reply(brain, idle)[1] == "script"  # invalid AI output still used a daily attempt
        assert chat.reply(brain, lambda: False)[0] is None
        assert client.messages.create.call_count == 2


if __name__ == "__main__":
    for check in (test_display_names, test_parse_request, test_picker, test_daily_cap, test_ai_path):
        check()
        print(f"OK: {check.__name__}")
