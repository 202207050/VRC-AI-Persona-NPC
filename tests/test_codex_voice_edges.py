r"""Offline checks. From Tools/voice: .venv\Scripts\python.exe -X utf8 tests/test_codex_voice_edges.py."""

import json
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cfg_bridge
import persona_ai
import persona_store
import smalltalk


STAMP = "2026.11.19 13:44:37 Debug      -  "


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def ai_wrote(store, name, impression="", note=""):
    """What the memory call (t20) stores: a stand-in answer through memory_finish."""
    p = store.impression(name)
    return store.memory_finish({"name": name, "impression": p["note"], "profile": p["profile"]},
                               {"changed": True, "profile": [], "impression": impression,
                                "notes": [note] if note else []})


def make_store(folder):
    personas = folder / "personas"
    personas.mkdir()
    for npc, name in persona_store.DEFAULT_NPC_NAMES.items():
        (personas / f"{npc}.md").write_text(f"- 이름: {name}\n", encoding="utf-8")
    return persona_store.Store(str(folder / "data"), str(personas / "kwonjun.md"))


def store_reload_checks(folder):
    store = make_store(folder)
    data = Path(store.data_dir)
    data.mkdir(parents=True)
    write_json(data / "settings.json", {
        "keep_history": True, "auto_update": False, "compact_minutes": 0, "compact_note_count": 7.0,
    })
    (data / "impressions.json").write_text("{broken", encoding="utf-8")
    write_json(data / "memory.json", {
        "notes": [None, "  오래된\n 메모  ", {"text": "  "}, {"text": "귤" * 120, "by": "ai", "at": "fixed"}],
    })
    history = []
    for i in range(persona_store.MAX_HISTORY + 2):
        history += [{"role": "user", "content": f"질문 {i}"},
                    {"role": "assistant", "content": f"대답 {i}"}]
    history[1:1] = [None, {"role": "system", "content": "skip"}, {"role": "user", "content": 42}]
    history.append({"role": "user", "content": "미완료"})
    write_json(data / "history.json", history)

    loaded = persona_store.Store(store.root_dir, store.persona_path)
    assert loaded.settings["keep_history"] and not loaded.settings["auto_update"]
    assert loaded.settings["compact_minutes"] == persona_store.DEFAULT_SETTINGS["compact_minutes"]
    assert type(loaded.settings["compact_note_count"]) is int and loaded.settings["compact_note_count"] == 7
    assert loaded.players == {}, "Unreadable impressions must not prevent loading the other files"
    assert [(n["text"], n["by"]) for n in loaded.notes] == [
        ("오래된 메모", "user"), ("귤" * persona_store.MAX_NOTE_CHARS, "ai"),
    ]
    assert loaded.notes[1]["at"] == "fixed"
    assert len(loaded.history) == 2 * persona_store.MAX_HISTORY and loaded.history[0]["content"] == "질문 2"
    assert loaded.history[-1]["content"] == f"대답 {persona_store.MAX_HISTORY + 1}"
    assert not loaded.for_npc("itoa").history

    loaded.apply({"settings": {"keep_history": False}})
    assert not (data / "history.json").exists()
    assert len(loaded.history) == 2 * persona_store.MAX_HISTORY, "The current session still has its history"
    assert not persona_store.Store(store.root_dir, store.persona_path).history


def store_validation_and_capacity_checks(folder):
    store = make_store(folder)
    try:
        store.apply({"persona": "- 이름: 바뀐 이름", "settings": {"auto_update": False, "compact_minutes": True}})
    except ValueError:
        pass
    else:
        raise AssertionError("A boolean is not a valid numeric compaction setting")
    assert store.settings["auto_update"] is True and store.npc_name() == "kwonjun"
    assert not Path(store.data_dir, "settings.json").exists()
    assert Path(store.persona_path).read_text(encoding="utf-8") == "- 이름: kwonjun\n"

    notes = [{"text": f"수동 {i}", "by": "user"} for i in range(persona_store.MAX_NOTES - 2)]
    notes[1:1] = [{"text": "오래된 자동 메모", "by": "ai"}, {"text": "최근 자동 메모", "by": "ai"}]
    store.apply({"notes": notes})
    assert store.add_note("한도 초과")
    store.after_reply("손님", "안녕", "응")
    duplicate = ai_wrote(store, "손님", note="  수동   0 ")
    assert duplicate["notes"] == [] and len(store.notes) == persona_store.MAX_NOTES
    store.after_reply("손님", "또 안녕", "응")
    changed = ai_wrote(store, "손님", note="새 자동 메모")
    assert changed["notes"] == ["새 자동 메모"]
    expected = [n["text"] for n in notes if n["text"] != "오래된 자동 메모"] + ["새 자동 메모"]
    loaded = persona_store.Store(store.root_dir, store.persona_path)
    assert [n["text"] for n in loaded.notes] == expected
    assert loaded.notes[-1]["by"] == "ai" and loaded.impression("손님")["talks"] == 2


def cfg_dictation_checks(folder):
    store = make_store(folder)
    now = [1000.0]
    bridge = cfg_bridge.CfgBridge(store, clock=lambda: now[0])
    store.set_speaker("귤 손님")

    def dictate(target, npc, source):
        req = json.dumps({"op": "dictate", "target": target, "npc": npc})
        return bridge.handle_line(source, "local:test", STAMP + f"[W04AI] CFG_SET {req} utc=1\n")

    assert dictate("note_add", "kwonjun", "a")
    previous = bridge.dictate_target()
    assert previous["npc"] == "kwonjun"
    assert dictate("impression", "itoa", "b")
    active = bridge.dictate_target()
    assert active["player"] == "귤 손님" and active["npc"] == "itoa"
    assert bridge.dictate_apply("이전 녹음", expected=previous)
    assert bridge.dictate_target() == active and not store.notes
    assert json.loads(bridge.payload("kwonjun"))["dictate"] is None
    assert json.loads(bridge.payload("itoa"))["dictate"] == {"target": "impression", "player": "귤 손님"}
    assert not bridge.apply({"op": "dictate", "target": "none", "npc": "kwonjun"})
    bridge.world_changed("a")
    assert bridge.dictate_target() == active
    assert not bridge.dictate_apply("새 녹음", expected=active)
    assert store.for_npc("itoa").impression("귤 손님")["note"] == "새 녹음" and not store.players
    assert bridge.dictate_target() is None

    assert dictate("note_add", "itoa", "b")
    expiring = bridge.dictate_target()
    now[0] = expiring["until"]
    assert bridge.dictate_apply("시간 초과", expected=expiring) and bridge.dictate_target() is None
    assert not store.for_npc("itoa").notes


def cfg_payload_checks(folder):
    store = make_store(folder)
    bridge = cfg_bridge.CfgBridge(store)
    store.set_speaker("현재 화자")
    store.after_reply("오래된 손님", "옛" * cfg_bridge.MAX_CFG_BYTES, "답" * cfg_bridge.MAX_CFG_BYTES)
    store.after_reply("새 손님", "방금 한 말", '{"lines":{"ko":"방금 한 답"}}')
    initial = bridge.payload("kwonjun")
    body = json.loads(initial)
    assert len(initial) <= cfg_bridge.MAX_CFG_BYTES and bridge.payload("kwonjun") == initial
    assert body["history"] == [{"who": "user", "text": "방금 한 말"}, {"who": "npc", "text": "방금 한 답"}]
    assert body["players"][0] == {"name": "현재 화자", "note": "", "locked": False}

    assert not store.set_impression("현재 화자", "화자의 인상")
    for i in range(12):
        assert not store.set_impression(f"손님 {i}", "귤" * persona_store.MAX_IMPRESSION_CHARS)
    pressured = bridge.payload("kwonjun")
    updated = json.loads(pressured)
    assert len(pressured) <= cfg_bridge.MAX_CFG_BYTES and updated["rev"] == body["rev"] + 1
    assert updated["history"] == []
    assert 1 < len(updated["players"]) < len(store.players)
    assert updated["players"][0]["name"] == "현재 화자" and updated["players"][0]["note"] == "화자의 인상"
    assert "persona_cut" not in updated and updated["persona"] == store.persona_text()
    assert json.loads(bridge.payload("itoa"))["history"] == [] and len(store.history) == 4


def clamp_steps_checks(_folder):
    look = {"do": "look_at", "target": "sofa", "gesture": "nod"}
    walk = {"do": "move_to", "target": "bridge_view", "gesture": "none"}
    assert persona_ai.clamp_steps((look,)) == [], "The JSON steps container must be a list"
    assert persona_ai.clamp_steps([look, walk], actions=[]) == [], "An empty allowlist permits no actions"
    assert persona_ai.clamp_steps([look, walk], actions=["move_to"]) == [walk]
    assert persona_ai.clamp_steps([{"do": "wait"}, {"do": ["move_to"]}]) == [
        {"do": "wait", "target": "speaker", "gesture": "none"},
    ]

    tv = {"do": "tv_left", "target": "tv", "gesture": "wave"}
    raw = [tv, look, {"do": "tv_reset"}, walk]
    left = {"do": "tv_left", "target": "none", "gesture": "none"}
    reset = {"do": "tv_reset", "target": "none", "gesture": "none"}
    # t128: four steps fit (the world's MaxSteps), so the look is no longer squeezed out.
    assert persona_ai.clamp_steps(raw) == [look, walk, left, reset]
    assert persona_ai.clamp_steps([tv, left]) == [left]
    assert persona_ai.clamp_steps([tv, reset, {"do": "tv_raise"}]) == [left, reset]
    assert raw[0] == {"do": "tv_left", "target": "tv", "gesture": "wave"}
    at_limit = [None] * (persona_ai.MAX_STEPS - 1) + [look, walk]
    assert persona_ai.clamp_steps(at_limit) == [look, walk], "Invalid entries do not take a slot"

    assert persona_ai.clamp_steps([{"do": "mirror", "action": "hq"}, {"do": "night_mode", "action": "30"}]) == [
        {"do": "mirror", "target": "none", "gesture": "none", "action": "hq"},
        {"do": "night_mode", "target": "none", "gesture": "none", "action": "30"},
    ]
    assert persona_ai.clamp_steps([{"do": "mirror", "action": "bright"}, {"do": "night_mode", "action": "150"},
                                   {"do": "night_mode", "action": "abc"}]) == []

    closer = {"do": "tv_closer_max", "target": "none", "gesture": "none"}
    size_max = {"do": "tv_size_max", "target": "none", "gesture": "none"}
    fill = persona_ai.fill_missing_tv
    heard = "TV 초기화하고 최대한 당겨줘"
    said = "원래대로 돌리고 최대한 가까이 당길게!"
    assert fill([reset], heard, said) == [reset, closer]
    assert fill([closer], heard, said) == [reset, closer]
    assert fill([closer, reset], heard, said) == [closer, reset]
    assert fill([size_max], "TV 사이즈 최대로 하고 TV 최대한 당겨줘", "최대로 키우고 최대한 당길게!") == [size_max, closer]
    assert fill([reset], heard, "원래대로 돌릴게!") == [reset]
    assert fill([reset], "TV 초기화해줘", said) == [reset]
    assert fill([closer], "TV 원상태로 돌리고 최대한 당겨줘", "원상태로 돌리고 최대한 가까이 당길게!") == [reset, closer]
    # t128: four steps, so the look stays; with all four taken the look makes room for the TV step.
    assert fill([look, walk, reset], heard, said) == [look, walk, reset, closer]
    assert fill([look, walk, walk, reset], heard, said) == [walk, walk, reset, closer]
    pause = {"do": "tv_pause", "target": "none", "gesture": "none"}
    assert fill([pause, reset], "TV 일시정지하고 원래대로 돌려줘", "일시정지하고 원래대로 돌릴게!") == [pause, reset]
    # "원래대로" about a pose or an outfit must not send the NPC to the TV
    assert fill([], "자세 원래대로 해", "원래대로 설게!") == []
    assert fill([look], "옷 원래대로 입어", "원래대로 입을게!") == [look]
    assert fill([], "화면 원래대로 해줘", "원래대로 돌릴게!") == [reset]


def clamp_reply_checks(_folder):
    blank = persona_ai._clamp_reply({"emotion": "unknown", "intensity": None, "lines": None, "steps": {}})
    assert blank == {"emotion": "neutral", "intensity": 0.5, "steps": [], "lines": {"ko": ""}, "line": ""}
    normalized = persona_ai._clamp_reply({
        "emotion": "happy", "intensity": "0.786", "lines": {"ko": "  귤\t1000개\n  왔어!  "},
    })
    assert normalized["emotion"] == "happy" and normalized["intensity"] == 0.79
    assert normalized["line"] == normalized["lines"]["ko"] == "귤 1,000개 왔어!"
    for value, expected in [("-0.1", 0.0), ("1.01", 1.0), ("invalid", 0.5)]:
        assert persona_ai._clamp_reply({"intensity": value})["intensity"] == expected, value

    limit = persona_ai.MAX_LINE_CHARS + 20
    exact = "🍊" * limit
    assert persona_ai._clamp_reply({"lines": {"ko": exact}})["line"] == exact
    long_after_formatting = "가" * (limit - 4) + "1000"
    shortened = persona_ai._clamp_reply({"lines": {"ko": long_after_formatting}})
    assert shortened["line"] == shortened["lines"]["ko"] == "가" * persona_ai.MAX_LINE_CHARS + "…"


def scripted_dialog(label):
    return [{"npc": "kwonjun", "line": f"{label} 이야기"}, {"npc": "itoa", "line": f"{label} 대답"}]


def smalltalk_pool_checks(folder):
    path = folder / "smalltalk.json"
    pair = ",".join(persona_store.NPC_IDS)
    first = scripted_dialog("첫째")
    second = scripted_dialog("둘째")
    invalid = [scripted_dialog("역순")[::-1], [{"npc": "kwonjun", "line": "홀로"}],
               [{"npc": "kwonjun", "line": " "}, second[1]]]
    with patch.object(smalltalk, "SCRIPT", path):
        write_json(path, {pair: invalid + [first]})
        one = smalltalk.SmallTalk(None)
        returned = one.scripted()
        assert [entry["line"] for entry in returned] == [entry["line"] for entry in first]
        returned[0]["line"] = "바뀐 임시 값"
        assert one.scripted()[0]["line"] == first[0]["line"]

        write_json(path, {pair: [first, second]})
        two = smalltalk.SmallTalk(None)
        chosen = [two.scripted()[0]["line"] for _ in range(4)]
        assert set(chosen) == {first[0]["line"], second[0]["line"]}
        assert chosen[:2] == chosen[2:] and chosen[0] != chosen[1]

        write_json(path, {pair: invalid})
        try:
            smalltalk.SmallTalk(None)
        except ValueError:
            pass
        else:
            raise AssertionError("An all-invalid script pool must be rejected")


def smalltalk_selection_checks(folder):
    from types import SimpleNamespace
    from unittest.mock import Mock

    store = make_store(folder)
    itoa = store.for_npc("itoa")
    itoa.set_smalltalk_settings({"ratio": 1.0, "daily_cap": 2})
    path = folder / "smalltalk.json"
    write_json(path, {",".join(persona_store.NPC_IDS): [scripted_dialog("준비")]})
    with patch.object(smalltalk, "SCRIPT", path):
        chat = smalltalk.SmallTalk(store)
    brain = SimpleNamespace(client=object(), smalltalk=Mock(side_effect=RuntimeError("offline fake")))
    idle = lambda: True

    with patch.object(persona_store.time, "strftime", return_value="2026-11-19"):
        assert chat.reply(brain, lambda: False) == (None, "script")
        brain.smalltalk.assert_not_called()
        fallback, source = chat.reply(brain, idle)
        assert source == "fallback" and fallback["smalltalk"] is True
        assert fallback["dialog"][0]["line"] == "준비 이야기"
        brain.smalltalk.assert_called_once_with([store, itoa])

        # Speech starts while the second AI call is in flight. Its result must not be queued.
        idle_now = [True]

        def respond(_stores):
            idle_now[0] = False
            return scripted_dialog("AI")

        brain.smalltalk.side_effect = respond
        assert chat.reply(brain, lambda: idle_now[0]) == (None, "ai")
        scripted, source = chat.reply(brain, idle)
        assert source == "script" and scripted["dialog"] == fallback["dialog"]
        assert brain.smalltalk.call_count == 2 and not store.history and not itoa.history


def affection_store_checks(folder):
    store = make_store(folder)
    itoa = store.for_npc("itoa")
    phrase = persona_store.AFFECTION_PHRASES["wave"]
    prefix = "귤" * persona_store.MAX_IMPRESSION_CHARS

    # t20: a full impression no longer blocks it: the wave is kept next to the impression, never inside it
    assert not itoa.set_impression("손님", prefix)
    assert itoa.add_affection("  손님  ", "wave") == phrase
    assert itoa.impression("손님")["note"] == prefix and itoa.impression("손님")["touch_recorded"] == ["wave"]
    assert itoa.add_affection("손님", "wave") == "", "each kind once per player"
    assert not itoa.set_impression("손님", "새 인상")
    assert itoa.add_affection("손님", "wave") == "" and itoa.impression("손님")["touch_recorded"] == ["wave"]
    assert phrase in itoa.prompt_tail("손님") and "새 인상" in itoa.prompt_tail("손님")
    assert itoa.impression("손님")["talks"] == 0 and "손님" not in store.players
    # an older file with the phrase inside the impression: moved next to it on load
    (folder / "old").mkdir()
    old = make_store(folder / "old")
    Path(old.data_dir).mkdir(parents=True)
    write_json(Path(old.data_dir) / "impressions.json",
               {"players": {"A": {"note": "다정함. " + persona_store.AFFECTION_PHRASES["pat"], "talks": 1}}})
    loaded = persona_store.Store(old.root_dir, old.persona_path)
    assert loaded.players["A"]["note"] == "다정함." and loaded.players["A"]["touch_recorded"] == ["pat"]
    assert loaded.players["A"]["profile"] == [], "no profile in an older file: an empty list"


def affection_log_checks(folder):
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    import persona_npc_app as app

    store = make_store(folder)
    file = folder / "output_log_test.txt"
    file.touch()
    stop = threading.Event()
    logger = Mock()
    name = "귤  팬 npc=kwonjun"
    long_name = "나" * (persona_store.MAX_NAME_CHARS + 5)
    events = [
        "[W04AI] AFFECTION type=wave npc=itoa by=" + name + "  ",
        "[W04AI] AFFECTION type=hug npc=unknown by=" + long_name,
        "[W04AI] AFFECTION type=bite npc=itoa by=잘못된 유형",
        "[W04AI] AFFECTION type=pat by=NPC 없음",
        "[W04AI] AFFECTION type=wave npc=itoa by=",
        "[Behaviour] Joining wrld_1234:123",
        "[W04AI] AFFECTION type=pat npc=itoa by=허용 안 된 월드",
        "[Behaviour] Joining local:test",
        "[W04AI] AFFECTION type=pat npc=itoa by=복귀한 손님",
    ]

    def lines():
        for event in events:
            yield STAMP + event + "\r\n"
        stop.set()

    tail = app.LogTail(str(file))
    tail.world = "local:test"
    tail.npc_state["itoa"] = True
    tail.lines = lines
    tail.close = Mock(wraps=tail.close)
    bridge = cfg_bridge.CfgBridge(store, log=logger)
    watcher_threads = SimpleNamespace(Lock=threading.Lock, Thread=Mock())
    with patch.object(app, "recent_output_logs", return_value=[str(file)]), \
            patch.object(app, "LogTail", return_value=tail), patch.object(app, "log", logger), \
            patch.object(app, "threading", watcher_threads):
        app.watch_log(str(folder), None, None, False, store=store, bridge=bridge, stop=stop, listen_mode="ptt")

    itoa = store.for_npc("itoa")
    assert {who: data["touch_recorded"] for who, data in itoa.players.items()} == {
        "귤 팬 npc=kwonjun": ["wave"], "복귀한 손님": ["pat"],
    }
    assert {who: data["touch_recorded"] for who, data in store.players.items()} == {
        long_name[:persona_store.MAX_NAME_CHARS]: ["hug"],
    }
    assert all(data["note"] == "" for data in [*itoa.players.values(), *store.players.values()])
    affection = [call.args[1] for call in logger.call_args_list if call.args[0] == "AFFECTION"]
    noted = [detail.split("\t")[:2] for detail in affection if detail.endswith("\tnoted=1")]
    assert noted == [["type=wave", "npc=itoa"], ["type=hug", "npc=kwonjun"], ["type=pat", "npc=itoa"]]
    assert all("type=bite" not in detail and "NPC 없음" not in detail for detail in affection)
    tail.close.assert_called_once()


def main():
    checks = (
        store_reload_checks,
        store_validation_and_capacity_checks,
        cfg_dictation_checks,
        cfg_payload_checks,
        clamp_steps_checks,
        clamp_reply_checks,
        smalltalk_pool_checks,
        smalltalk_selection_checks,
        affection_store_checks,
        affection_log_checks,
    )
    failed = 0
    with tempfile.TemporaryDirectory(prefix="w04_codex_voice_") as tmp:
        for check in checks:
            folder = Path(tmp) / check.__name__
            folder.mkdir()
            try:
                with patch.object(persona_store, "HERE", str(folder)):
                    check(folder)
            except Exception:
                failed += 1
                print(f"FAIL {check.__name__}\n{traceback.format_exc()}", flush=True)
            else:
                print(f"PASS {check.__name__}", flush=True)
    print(f"PASS {len(checks) - failed} / FAIL {failed}")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
