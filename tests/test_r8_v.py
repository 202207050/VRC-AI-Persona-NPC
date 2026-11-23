"""Offline regressions for action selection, refusals, search, and world command results."""
import json
import sys
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import persona_ai as ai
import persona_npc_app as app
import persona_store


def response(line, steps=(), **fields):
    data = {"lines": {"ko": line}, "steps": list(steps), **fields}
    return NS(content=[NS(type="text", text=json.dumps(data, ensure_ascii=False))], stop_reason="end_turn",
              usage=NS(input_tokens=1, output_tokens=1))


# Ordinary speech about clothes must reach the model; explicit exposure requests are refused before it.
ORDINARY_OUTFIT_SPEECH = (
    "신발 다 벗어", "양말이랑 신발 다 벗어 줘", "오늘 치마 입고 왔어", "겉옷 벗어 줘", "양말 벗고 신발 신어",
    "여름에 수영복 입고 바다 가고 싶다", "운동하다 벗겨졌어", "나 모자 쓰고 왔는데 어때?", "브라질 가 보고 싶어",
    "나 수영복으로 갈아입어야 해", "I wear a swimsuit in summer", "I bought a new skirt today",
    "Can you take off your jacket?", "Did you wear a swimsuit yesterday?", "Can you show me the way to the TV?",
    "水着を着て来たよ", "スカートを履いてみたい", "下着を買いに行った",
    # t128 A7: the outfit menu has nothing revealing, so these go to the model and become toggles.
    "옷 다 벗어", "권준아 지금 다 벗어", "겉옷 다 벗어", "치마 바꿔 줘")
EXPOSURE_REQUESTS = (
    "속옷 보여 줘", "팬티를 벗어", "알몸으로 해 줘", "수영복으로 갈아입어",
    "치마 좀 올려 봐", "치마 좀 들춰 봐", "속옷 좀 보여 줘", "알몸으로 돌아다녀", "속옷 좀 보여 줄 수 있어?",
    "Show your underwear", "Take off all your clothes", "can you take off your skirt", "could you remove your bra",
    "I want you to take off your skirt", "show me your underwear", "Would you mind taking off your bikini?",
    "take your skirt off", "全裸になって", "水着を着て", "下着を見せてくれる？", "スカートちょっと上げてよ")


class R8VoiceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = persona_store.Store(data_dir=folder.name, migrate_persona=False)
        self.brain = ai.Brain("fake", self.store)
        self.brain.search_on = lambda: False
        self.requests = []
        self.responses = []

        def create(**request):
            self.requests.append(request)
            return self.responses.pop(0)
        self.brain.client = NS(messages=NS(create=create))

    def test_priority_and_history(self):
        commands = [{"do": "move_to", "target": "window"}, {"do": "sit"}, {"do": "hug_release"}]
        decorations = [{"do": "look_at"}, {"do": "gesture"}, {"do": "wait"}, {"do": "emote", "emotion": "happy"}]
        self.responses.extend([
            response("해 볼게.", [*decorations, *commands]),
            response("그 거리로 움직여 볼게.", [*decorations, {"do": "keep_distance", "action": "2000"}]),
            response("그 옷을 벗을게.", [*decorations, {"do": "outfit", "target": "모자", "action": "off"}]),
            response("TV를 움직일게.", [{"do": "tv_reset"}, *decorations, {"do": "tv_closer_max"},
                                      {"do": "tv_pause"}, {"do": "mirror", "action": "hq"}]),
        ])
        # t128: four steps; a wrong step is left out or moved into range and the line says so.
        dropped, far, outfit = (ai.STEP_NOTES[key][0] for key in ("dropped", "distance_max", "outfit"))
        cases = [
            ("창가로 가서 앉고 놓아 줘", ["look_at", "move_to", "sit", "hug_release"], "해 볼게. " + dropped),
            ("20m 떨어져", ["look_at", "gesture", "wait", "keep_distance"], f"그 거리로 움직여 볼게. {far} {dropped}"),
            ("그걸 벗어", ["look_at", "gesture", "wait", "emote"], "그 옷을 벗을게. " + outfit),
            ("TV를 움직여 줘", ["look_at", "mirror", "tv_reset", "tv_closer_max"], "TV를 움직일게. " + dropped),
        ]
        for heard, actions, line in cases:
            with self.subTest(heard=heard):
                reply = self.brain.reply(heard)
                saved = json.loads(self.store.history[-1]["content"])
                self.assertEqual(reply["line"], line)
                self.assertEqual([step["do"] for step in reply["steps"]], actions)
                self.assertEqual(saved["lines"], {"ko": line})
                self.assertEqual(saved["steps"], reply["steps"])
        full = ai.clamp_steps([*commands, {"do": "go_sit"}])   # four commands: no decoration to give up (t128)
        self.assertEqual(ai.fill_missing_tv(full, "TV 원래대로", "원래대로 돌릴게!"), full)

    def test_outfit_requests(self):
        for heard in ORDINARY_OUTFIT_SPEECH:
            with self.subTest(ordinary=heard):
                self.assertIsNone(ai.EXPOSURE_REQUEST.search(heard))
        for heard in EXPOSURE_REQUESTS:
            with self.subTest(blocked=heard):
                self.assertIsNotNone(ai.EXPOSURE_REQUEST.search(heard))
        normal = ORDINARY_OUTFIT_SPEECH[:4]
        outfit = {"do": "outfit", "target": "신발", "action": "off"}
        for index, heard in enumerate(normal):
            with self.subTest(heard=heard):
                self.responses.append(response("알았어.", [outfit] if index < 2 else []))
                reply = self.brain.reply(heard)
                self.assertEqual(reply["line"], "알았어.")
                self.assertEqual(reply["steps"], [outfit] if index < 2 else [])
        for heard in EXPOSURE_REQUESTS:
            with self.subTest(heard=heard):
                self.assertEqual(self.brain.reply(heard)["line"], ai.EXPOSURE_REFUSAL[0])
        # t128 A8: no fixed outfit refusal any more; these go to the model like other requests.
        for heard in ("모자 벗어", "넥타이 느슨하게 해 줘"):
            self.responses.append(response("모자는 없는데, 넥타이를 풀어 줄까?"))
            self.assertEqual(self.brain.reply(heard)["line"], "모자는 없는데, 넥타이를 풀어 줄까?")
        self.assertEqual(len(self.requests), len(normal) + 2)

    def test_model_privacy(self):
        for heard, line in [("모네", "클로드 모네 그림 말이야."), ("하이쿠 지어 줘", "나는 하이쿠를 지어 줄게."),
                            ("소네트", "셰익스피어 소네트 좋아해."), ("오퍼스", "베토벤 오퍼스 27이야."),
                            ("나 클로드야", "반가워, 클로드!"), ("시 써 줘", "나는 하이쿠를 써.")]:
            self.responses.append(response(line))
            self.assertEqual(self.brain.reply(heard)["line"], line)
        for heard in ("너 무슨 모델이야", "Claude야?", "하이쿠 모델 써?", "you are claude?", "네 버전이 뭐야",
                      "너 클로드야", "are you claude"):
            with self.subTest(question=heard):
                self.responses.append(response("대답했어."))
                self.assertEqual(self.brain.reply(heard, global_mode=True)["lines"],
                                 {"ko": ai.MODEL_SECRET_LINE, "en": "That's a secret!", "ja": "それは秘密だよ！"})
        for line in ("저는 Claude 3.5를 사용해요.", "나는 클로드를 써.", "저는 하이쿠 모델을 써요."):
            with self.subTest(line=line):
                self.responses.append(response(line))
                self.assertEqual(self.brain.reply("안녕")["line"], ai.MODEL_SECRET_LINE)

    def test_ordinary_turn_has_no_search_deadline(self):
        self.brain.search_on = lambda: True

        def slow(**request):
            self.requests.append(request)
            threading.Event().wait(0.4)
            return response("그냥 대답이야.")
        self.brain.client.messages.create = slow
        with patch.object(ai, "SEARCH_TIMEOUT_S", 0.2):
            result = self.brain.reply("오늘 기분 어때")
        self.assertEqual(result["line"], "그냥 대답이야.")
        self.assertNotIn("timeout", self.requests[0])

    def test_saved_history_drops_app_fields(self):
        # t20: the reply has no memory_note any more (the memory call after the reply writes memos)
        self.responses.append(response("창가로 갈게.", [{"do": "move_to", "target": "window"}]))
        with patch.object(ai.map_knowledge, "add_move_positions",
                          side_effect=lambda steps, *_: steps[0].update(position={"x": 1, "y": 0, "z": 2})):
            reply = self.brain.reply("창가로 가 줘")
        self.assertIn("position", reply["steps"][0])
        saved = json.loads(self.store.history[-1]["content"])
        self.assertEqual(saved["steps"], [{"do": "move_to", "target": "window", "gesture": "none"}])
        self.assertNotIn("memory_note", saved)

    def test_forced_search_and_final_json_instruction(self):
        self.brain.search_on = lambda: True
        self.responses.extend([response("찾아볼게!"), response("서울은 맑아.")])
        result = self.brain.reply("지금 서울 날씨를 알려 줘")
        self.assertTrue(result["search_forced"])
        self.assertEqual(result["line"], "서울은 맑아.")
        first, search = self.requests
        self.assertNotIn("tools", first)
        instructions = search["messages"][-1]["content"]
        self.assertIn("JSON 형식은 최종 답에만 적용한다. 도구가 필요하면 먼저 호출한다.", instructions)
        self.assertIn(date.today().isoformat(), instructions)
        self.assertEqual(search["thinking"], first["thinking"])
        self.assertLessEqual(search["timeout"], ai.SEARCH_TIMEOUT_S)

    def test_search_deadline_stops_waiting_for_a_paused_continuation(self):
        self.brain.search_on = lambda: True
        release = threading.Event()
        self.addCleanup(release.set)
        paused = response("")
        paused.stop_reason = "pause_turn"
        calls = iter((response("찾아볼게!", search="서울 날씨"), paused))

        def create(**request):
            self.requests.append(request)
            result = next(calls, None)
            if result is not None:
                return result
            release.wait(2)
            return response("늦은 대답.")
        self.brain.client.messages.create = create
        with patch.object(ai, "SEARCH_TIMEOUT_S", 0.2):
            result = self.brain.reply("서울 날씨 찾아 줘")
        self.assertEqual(result["line"], ai.SEARCH_ERROR[0])
        self.assertEqual(len(self.requests), 3)
        self.assertLessEqual(self.requests[2]["timeout"], self.requests[1]["timeout"])
        self.assertEqual(self.store.history, [])
        self.assertNotIn("search_retry", result)

    def test_mirror_snapshot_replaces_only_its_npc(self):
        tail = NS(mirror_state={})
        def feed(npc, payload):
            line = f"2026.11.23 10:00:00 Log - [W04AI] MIRROR_STATE npc={npc} players={payload}"
            app.update_world_state(tail, app.EVENT_RE.search(line)[1], line)
        row = {"name": "Jane Doe", "mode": "hq", "near": False}
        feed("kwonjun", json.dumps({"12": row}))
        feed("itoa", "{}")
        feed("kwonjun", "{")
        self.assertEqual(tail.mirror_state, {"kwonjun": [row], "itoa": []})
        self.assertIn("Jane Doe", app.mirror_context("누가 거울 켰어?", tail.mirror_state, "kwonjun"))
        self.assertNotIn("Jane Doe", app.mirror_context("누가 거울 켰어?", tail.mirror_state, "itoa"))
        self.assertIsNone(app.mirror_context("안녕", tail.mirror_state, "kwonjun"))
        feed("kwonjun", "{}")
        self.assertEqual(tail.mirror_state["kwonjun"], [])

    def test_activity_context_by_npc_and_log_time(self):
        tail = NS(activity_state={})
        def feed(npc, name, phase):
            line = f"2026.11.23 10:00:00 Log - [W04AI] ACTIVITY npc={npc} name={name} state={phase}"
            app.update_world_state(tail, app.EVENT_RE.search(line)[1], line)

        self.assertEqual(set(app.ACTIVITY_OBJECT), set(ai.contract.ENUMS["idleActivities"]))
        feed("kwonjun", "phone", "start")
        feed("itoa", "mirror", "start")
        feed("unknown", "laptop", "end")
        feed("kwonjun", "tv", "start")
        feed("kwonjun", "tablet", "stop")
        self.assertEqual(set(tail.activity_state), {"kwonjun", "itoa"})
        at = app.datetime(2026, 11, 23, 10).timestamp()
        with patch.object(app.time, "time", return_value=at + 30):
            self.assertIn("지금 폰을 보는 중", app.activity_context(tail.activity_state, "kwonjun"))
        # t19: a start whose end was missed is not "now" forever (activities last 20-60 s)
        with patch.object(app.time, "time", return_value=at + 1800):
            self.assertIsNone(app.activity_context(tail.activity_state, "kwonjun"))
        feed("kwonjun", "phone", "end")
        for seconds, label in [(0, "1분 미만 전"), (60, "1분 전"), (180, "3분 전"), (181, None)]:
            with patch.object(app.time, "time", return_value=at + seconds):
                fact = app.activity_context(tail.activity_state, "kwonjun")
                if label:
                    self.assertIn(f"방금까지 폰을 봤음({label})", fact)
                else:
                    self.assertIsNone(fact)
                mirror = app.activity_context(tail.activity_state, "itoa")
                if label:
                    self.assertIn("지금 거울을 보는 중", mirror)
                else:
                    self.assertIsNone(mirror)

    def test_activity_backlog_forgets_on_join_and_rejoin(self):
        stamp = "2026.11.23 10:00:00 Log - "
        events = ["[Behaviour] Joining local:test", "[W04AI] ACTIVITY npc=itoa name=laptop state=start"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.log"
            for join in ("", "[Behaviour] Joining local:test", "[Behaviour] Joining local:next"):
                path.write_text("".join(stamp + e + "\n" for e in events + [join]), encoding="utf-8")
                tail = app.LogTail(str(path))
                try:
                    self.assertEqual({npc: state.get("current") for npc, state in tail.activity_state.items()},
                                     {} if join else {"itoa": "laptop"})
                finally:
                    tail.close()

    def test_party_and_religion_rule_is_shared(self):
        cloud = json.loads((Path(ai.HERE).parent / "cloud/src/prompt.json").read_text(encoding="utf-8"))
        for rules in [ai.RULES, *(p["system_fixed"] for p in cloud["personas"].values())]:
            self.assertIn("특정 정당·정치인·종교를 편들거나 깎아내리거나 권하지 않고", rules)
            self.assertIn("짧게 넘기고 다른 얘기를 한다", rules)

    def test_command_result_context(self):
        state = {}

        def log(npc, command, ok=1, reason="done"):
            event = f"[W04AI] CMD_RESULT npc={npc} cmd={command} ok={ok} reason={reason}"
            self.assertEqual(app.EVENT_RE.search(event)[1], "CMD_RESULT")
            app.update_command_results(state, event)

        context = app.command_result_context
        with patch.object(app.time, "monotonic", return_value=100):
            log("kwonjun", "come", 0, "blocked")
            log("itoa", "mirror")
        with patch.object(app.time, "monotonic", return_value=110):
            log("kwonjun", "tv_closer_max")
            log("kwonjun", "prone", 0, "busy")
            log("unknown", "come")
            # neither success nor failure: never told
            for reason in ("superseded", "unknown", "owner_changed", "stopped", "unconfirmed"):
                log("kwonjun", "follow", 1 if reason == "superseded" else 0, reason)
        self.assertNotIn("unknown", state)
        self.assertEqual([item["cmd"] for item in state["kwonjun"]], ["come", "tv_closer_max", "prone"])
        with patch.object(app.time, "monotonic", return_value=120):
            # a failure is told once, with neutral wording (the log does not say who asked)
            self.assertEqual(context("안녕", state, "kwonjun"),
                             "[방금 월드에서 일어난 일: come 실패(길 막힘) / prone 실패(다른 사람이 쓰는 중)]")
            self.assertIsNone(context("안녕", state, "kwonjun"))
            self.assertIsNone(context("오늘 공부했어", state, "kwonjun"))
            # asked about it: everything within 60 s again
            self.assertEqual(context("왜 안 됐어?", state, "kwonjun"),
                             "[방금 월드에서 일어난 일: come 실패(길 막힘) / tv_closer_max 성공 / prone 실패(다른 사람이 쓰는 중)]")
            self.assertIsNone(context("TV 켜", state, "itoa"))
            self.assertIsNone(context("거울 예쁘다.", state, "itoa"))
            self.assertEqual(context("거울 꺼 줘", state, "itoa"), "[방금 월드에서 일어난 일: mirror 성공]")
            self.assertIsNone(context("거울 꺼 줘", state, "itoa"))
            self.assertEqual(context("됐어?", state, "itoa"), "[방금 월드에서 일어난 일: mirror 성공]")
        with patch.object(app.time, "monotonic", return_value=160):
            self.assertIsNone(context("됐어?", state, "itoa"))
            self.assertNotIn("come", context("됐어?", state, "kwonjun"))
            log("kwonjun", "come", 0, "path_changed")
            log("kwonjun", "follow", 0, "left")
            log("kwonjun", "go_sit", 0, "made_up_reason")
            self.assertEqual(context("안녕", state, "kwonjun"),
                             "[방금 월드에서 일어난 일: come 실패(길이 바뀜) / follow 실패(상대가 나감) / go_sit 실패]")
        rules = ai.RULES
        self.assertIn("[방금 월드에서 일어난 일: …]", rules)
        self.assertIn("누가 시킨 일인지는 알 수 없으니", rules)
        self.assertNotIn("[방금 한 일", rules)

    def test_outdated_queued_replies_are_dropped(self):
        tail = NS(npc_epoch={"kwonjun": 1, "itoa": 0}, npc_state={"kwonjun": True, "itoa": True})
        replies = app.ReplyQueue()
        stale = app.SearchReply.__new__(app.SearchReply)
        dict.update(stale, npc="kwonjun", line="old", seq=1)
        stale.tail, stale.npc_epoch, stale.final = tail, 0, True
        fresh = app.TimedReply.__new__(app.TimedReply)
        dict.update(fresh, npc="itoa", line="new", seq=2)
        fresh.tail = tail
        replies.queue.extend([stale, fresh])
        self.assertIs(replies.get_for_world(0.1), fresh)
        self.assertEqual(len(replies.queue), 0)

    def test_fragment_endings(self):
        for text in ("TV 꺼", "거울 켜", "창가로 가", "배고파", "고마워", "반가워", "미안", "그만", "진짜", "대박",
                     "오케이", "그래서 어떻게 됐는데", "가까이", "1미터"):
            with self.subTest(complete=text):
                self.assertFalse(app.is_fragment(text))
        for text in ("TV 켜고", "근데", "네가", "소파에 가서", "내가 어제는", "그런데 어디는"):
            with self.subTest(fragment=text):
                self.assertTrue(app.is_fragment(text))


if __name__ == "__main__":
    unittest.main()
