"""TODO 135 / 137 (offline): a slow AI server (shorter waits, one retry, "생각 중" again, a line that
says why) and TV steps nobody asked about, tv_level, the remote-control self description."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import persona_ai as ai
import persona_store


class APITimeoutError(Exception):        # same class name as the SDK's (ai._server_slow checks names)
    pass


class BadRequestError(Exception):
    status_code = 400


class OverloadedError(Exception):
    status_code = 529


def response(line, steps=()):
    data = {"lines": {"ko": line}, "steps": list(steps)}
    return NS(content=[NS(type="text", text=json.dumps(data, ensure_ascii=False))], stop_reason="end_turn",
              usage=NS(input_tokens=1, output_tokens=1))


class SlowServerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = persona_store.Store(data_dir=folder.name, migrate_persona=False)
        self.brain = ai.Brain("fake", self.store)
        self.brain.search_on = lambda: False
        self.requests, self.answers = [], []

        def create(**request):
            self.requests.append(request)
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.brain.client = NS(messages=NS(create=create))

    def test_a_try_never_cuts_a_normal_answer(self):
        # slowest normal call seen in voice_log.tsv (14.4 s) with room to spare; 10 s cut live eval answers
        self.assertGreaterEqual(ai.REPLY_TIMEOUT_S, 14.4 * 1.35)

    def test_timeout_is_retried_once_with_a_thinking_refresh(self):
        self.answers.extend([APITimeoutError("Request timed out."), response("응, 좋아.")])
        interim = []
        result = self.brain.reply("오늘 어땠어?", on_interim=interim.append)
        self.assertEqual(result["line"], "응, 좋아.")
        self.assertEqual(interim, [{"interim": "slow"}])
        self.assertNotIn("timeout", self.requests[0])                      # the client's REPLY_TIMEOUT_S
        self.assertEqual(self.requests[1], self.requests[0])
        self.assertEqual(result["calls"][0]["retry"], "APITimeoutError")
        self.assertNotIn("on_retry", result["usage"])

    def test_two_timeouts_say_the_server_was_slow(self):
        self.answers.extend([APITimeoutError("Request timed out."), APITimeoutError("Request timed out.")])
        result = self.brain.reply("오늘 어땠어?")
        self.assertEqual(result["line"], ai.VOICE_SLOW[0])
        self.assertTrue(result["error"].startswith("APITimeoutError"))
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.store.history, [])

    def test_busy_server_pauses_before_the_retry(self):
        self.answers.extend([OverloadedError("overloaded"), response("응.")])
        with patch.object(ai.time, "sleep") as sleep:
            self.assertEqual(self.brain.reply("안녕")["line"], "응.")
        sleep.assert_called_once_with(ai.RETRY_PAUSE_S)

    def test_other_errors_are_not_retried_and_keep_the_old_line(self):
        self.answers.append(BadRequestError("bad"))
        result = self.brain.reply("안녕")
        self.assertEqual(result["line"], ai.VOICE_ERROR[0])
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(result["error"].startswith("BadRequestError"))

    def test_server_slow(self):
        self.assertTrue(ai._server_slow(APITimeoutError()))
        self.assertTrue(ai._server_slow(OverloadedError()))
        self.assertFalse(ai._server_slow(BadRequestError()))
        self.assertFalse(ai._server_slow(ai.SearchDeadlineExceeded()))
        self.assertFalse(ai._server_slow(KeyError("x")))


TV_SIZE = [{"do": "tv_size_set", "inches": 200}]
LINE = {"ko": "좋아, 해 볼게."}


class HeardStepTests(unittest.TestCase):
    def test_number_without_a_target_is_not_a_tv_command(self):
        reply = ai._clamp_reply({"lines": {"ko": "37% 키울게!"}, "steps": TV_SIZE}, heard="0.37 일단 올려 고칠 거 좀 많았으니까")
        self.assertEqual(reply["steps"], [])
        self.assertEqual(reply["line"], ai.STEP_NOTES["no_target"][0])

    def test_only_tv_steps_go_and_the_note_needs_an_empty_command(self):
        # other steps and the model's line stay
        reply = ai._clamp_reply({"lines": LINE, "steps": [{"do": "go_sit"}, *TV_SIZE]}, heard="저기 앉아서 올려 봐")
        self.assertEqual([step["do"] for step in reply["steps"]], ["go_sit"])
        self.assertEqual(reply["line"], LINE["ko"])
        # not a request for a change: no note
        reply = ai._clamp_reply({"lines": LINE, "steps": TV_SIZE}, heard="오늘 37 정도 했어")
        self.assertEqual((reply["steps"], reply["line"]), ([], LINE["ko"]))

    def test_distance_is_not_the_screen_width(self):
        width = [{"do": "tv_size_set", "width_m": 1.5}]
        reply = ai._clamp_reply({"lines": {"ko": "가로 1.5m로 맞출게!"}, "steps": width}, heard="1.5m로 맞춰줘")
        self.assertEqual(reply["steps"], [{"do": "keep_distance", "target": "none", "gesture": "none", "action": "150"}])
        self.assertEqual(reply["line"], ai.STEP_NOTES["distance_guess"][0])
        reply = ai._clamp_reply({"lines": {"ko": "가로 3m로 맞출게."}, "steps": [{"do": "tv_size_set", "width_m": 3}]},
                                heard="가로 3m로 맞춰줘")
        self.assertEqual([step["do"] for step in reply["steps"]], ["tv_size_set"])
        reply = ai._clamp_reply({"lines": {"ko": "1.5m로 갈게."}, "steps": [{"do": "keep_distance", "action": "150"}]},
                                heard="1.5m로 맞춰줘")
        self.assertEqual((reply["steps"][0]["action"], reply["line"]), ("150", "1.5m로 갈게."))

    def test_after_a_tv_turn_the_number_is_for_the_tv(self):
        reply = ai._clamp_reply({"lines": {"ko": "20인치 더 키울게."}, "steps": TV_SIZE}, heard="20 더 올려",
                                tv_before=True)
        self.assertEqual([step["do"] for step in reply["steps"]], ["tv_size_set"])

    def test_recent_tv(self):
        said = {"role": "user", "content": "TV 크기 150으로"}
        answer = {"role": "assistant", "content": json.dumps({"lines": {"ko": "응"}, "steps": [{"do": "tv_size_set"}]})}
        chat = {"role": "assistant", "content": json.dumps({"lines": {"ko": "응"}, "steps": []})}
        self.assertTrue(ai.recent_tv([said, chat]))
        self.assertTrue(ai.recent_tv([{"role": "user", "content": "그거 해 줘"}, answer]))
        self.assertFalse(ai.recent_tv([{"role": "user", "content": "배고파"}, chat]))
        self.assertFalse(ai.recent_tv([]))
        self.assertTrue(ai.recent_tv([], ["TV 크기 바꿔 줘", "몇 인치로?"]))

    def test_level_is_not_a_full_reset(self):
        for heard in ("TV 각도 수평으로 맞춰", "TV 기울기 원래대로", "티비 똑바로 해 줘"):
            with self.subTest(heard=heard):
                reply = ai._clamp_reply({"lines": {"ko": "원래대로 할게."}, "steps": [{"do": "tv_reset"}]}, heard=heard)
                self.assertEqual([step["do"] for step in reply["steps"]], ["tv_level"])
        reply = ai._clamp_reply({"lines": {"ko": "초기화할게."}, "steps": [{"do": "tv_reset"}]}, heard="TV 각도 초기화")
        self.assertEqual([step["do"] for step in reply["steps"]], ["tv_reset"])
        reply = ai._clamp_reply({"lines": {"ko": "수평으로 맞출게."}, "steps": []}, heard="TV 수평으로 맞춰")
        self.assertEqual([step["do"] for step in reply["steps"]], ["tv_level"])

    def test_remote_control_is_in_the_rules_and_about_me(self):
        self.assertIn(ai.TV_HOW, ai.RULES)
        self.assertIn("tv_level", ai.ACTIONS)
        with tempfile.TemporaryDirectory() as folder:
            brain = ai.Brain("fake", persona_store.Store(data_dir=folder, migrate_persona=False))
            self.assertIn(ai.TV_HOW, brain.about_me())
            self.assertIn("펜 가져와", ai.RULES)


if __name__ == "__main__":
    unittest.main()
