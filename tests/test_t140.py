"""TODO 140 (+ the rest of 135), offline: error codes in the activity record, the log and /cfg, and the one
"잠깐만, 생각 중이야" line of a slow answer. No microphone, no Whisper, no API call.

    .venv\\Scripts\\python.exe -X utf8 tests\\test_t140.py
"""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import activity
import persona_ai as ai
import persona_npc_app as app
import persona_store


class APITimeoutError(Exception):        # the SDK's class names (persona_ai checks names, not classes)
    pass


class APIConnectionError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


class OverloadedError(Exception):
    status_code = 529


def response(line):
    data = {"lines": {"ko": line}, "steps": []}
    return NS(content=[NS(type="text", text=json.dumps(data, ensure_ascii=False))], stop_reason="end_turn",
              usage=NS(input_tokens=1, output_tokens=1))


class ErrorCodeTests(unittest.TestCase):
    def test_codes_by_cause(self):
        cases = [(APITimeoutError(), "A103"), (APIConnectionError(), "A104"), (OverloadedError(), "A105"),
                 (ValueError("no reply JSON in the model's text"), "A106"), (RuntimeError("empty line"), "A106"),
                 (KeyError("lines"), "A106"), (BadRequestError("bad"), "A107"), (TypeError("x"), "A107"),
                 (RuntimeError("refusal"), "A108"), (ai.SearchDeadlineExceeded(), "A110"),
                 (RuntimeError("search stalled"), "A111"), (RuntimeError("search did not finish"), "A111")]
        for error, code in cases:
            with self.subTest(error=type(error).__name__, code=code):
                self.assertEqual(ai.error_code(error), code)
                self.assertIn(code, activity.CODES)

    def test_stt_reasons(self):
        self.assertEqual(activity.stt_code("too_quiet rms=0.0012"), "V102")
        self.assertEqual(activity.stt_code("too_short held=120ms"), "V101")
        self.assertEqual(activity.stt_code("hallucination '시청해 주셔서 감사합니다'"), "V105")
        self.assertEqual(activity.stt_code("unclear_short"), "V106")
        self.assertEqual(activity.stt_code("ok rms=0.1 dur=1.00s"), "")

    def test_rows_carry_the_code_and_the_table_words(self):
        trail = activity.Turn("itoa", 3, start=0.0, sink=None)
        trail.add_code("A103", at=20.1)
        trail.add_code("A108", at=20.2)
        trail.add_code("V110", kind="skip", at=20.3)
        trail.add_code("A101", kind="think", at=8.0)
        act = trail.act()
        self.assertTrue(activity.check(act))
        self.assertEqual(act["rows"][0], {"d": 20100, "k": "error", "x": "AI 서버 응답 시간 초과", "c": "A103"})
        self.assertEqual([(row["k"], row["c"]) for row in act["rows"]],
                         [("error", "A103"), ("drop", "A108"), ("skip", "V110"), ("think", "A101")])

    def test_add_result_codes_the_failure_and_the_step_notes(self):
        trail = activity.Turn("kwonjun", 1, start=0.0, sink=None)
        trail.add_result({"error": "BadRequestError: secret detail", "code": "A107", "step_notes": ["outfit"]},
                         0.0, 1.0, step_text=lambda key: ai.STEP_NOTES[key][0])
        rows = trail.act()["rows"]
        self.assertEqual(rows[0], {"d": 1000, "k": "error", "x": "AI 응답 오류 · BadRequestError", "c": "A107"})
        self.assertEqual(rows[1], {"d": 1000, "k": "drop", "x": ai.STEP_NOTES["outfit"][0], "c": "W203"})
        self.assertNotIn("secret", json.dumps(rows, ensure_ascii=False))

    def test_recent_errors_keep_warn_and_error_only_newest_first(self):
        log = activity.RecentErrors()
        log.note("A108")                         # info: never kept
        log.note("A103")
        log.note("V107")
        log.note("A103")
        snap = log.snapshot()
        self.assertEqual([(item["c"], item["n"]) for item in snap], [("A103", 2), ("V107", 1)])
        self.assertRegex(snap[0]["t"], r"^\d\d:\d\d:\d\d$")
        for code in ("A102", "A103", "A104", "A105", "A106", "A107", "A110", "A111", "V102", "V107", "V109",
                     "V112", "V114"):
            log.note(code)
        self.assertEqual(len(log.snapshot()), activity.RECENT_ERRORS)
        self.assertEqual(log.snapshot()[0]["c"], "V114")

    def test_turn_rows_count_in_recent_errors(self):
        activity.recent_errors.clear()
        try:
            trail = activity.Turn("kwonjun", 1, sink=None)
            trail.add_code("A105")
            trail.add_code("V110", kind="skip")
            self.assertEqual([item["c"] for item in activity.recent_errors.snapshot()], ["A105"])
        finally:
            activity.recent_errors.clear()


class BrainTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = persona_store.Store(data_dir=folder.name, migrate_persona=False)
        self.brain = ai.Brain("fake", self.store)
        self.brain.search_on = lambda: False
        self.answers, self.delay = [], 0.0

        def create(**request):
            time.sleep(self.delay)
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.brain.client = NS(messages=NS(create=create))

    def test_failed_reply_has_a_code_but_the_line_has_none(self):
        self.answers.extend([APITimeoutError("Request timed out."), APITimeoutError("Request timed out.")])
        result = self.brain.reply("오늘 어땠어?")
        self.assertEqual((result["code"], result["line"]), ("A103", ai.VOICE_SLOW[0]))
        self.answers.append(RuntimeError("refusal"))
        result = self.brain.reply("안녕")
        self.assertEqual(result["code"], "A108")
        self.assertNotIn("error", result)
        self.assertEqual(self.brain.reply("속옷 보여 줘")["code"], "A109")

    def test_ok_reply_has_no_code(self):
        self.answers.append(response("응, 좋아."))
        self.assertNotIn("code", self.brain.reply("안녕"))

    def test_slow_answer_says_wait_once_before_the_answer(self):
        self.answers.append(response("응, 좋아."))
        self.delay = 0.3
        said, order = [], []

        def interim(r):
            said.append(r)
            order.append("interim")
        with patch.object(ai, "WAIT_INTERIM_S", 0.05):
            result = self.brain.reply("오늘 어땠어?", on_interim=interim)
        order.append("answer")
        self.assertEqual(result["line"], "응, 좋아.")
        self.assertEqual(order, ["interim", "answer"])
        self.assertEqual(len(said), 1)
        self.assertEqual(said[0]["interim"], "wait")
        self.assertIn((said[0]["line"],), [(lines[0],) for lines in ai.INTERIM_LINES["wait"]["kwonjun"]])
        self.assertEqual(said[0]["lines"], {"ko": said[0]["line"]})

    def test_quick_answer_and_for_me_checks_say_nothing(self):
        self.answers.extend([response("응."), response("응.")])
        said = []
        with patch.object(ai, "WAIT_INTERIM_S", 0.05):
            self.brain.reply("안녕", on_interim=said.append)
            time.sleep(0.15)                 # the timer was cancelled, it does not fire later
            self.delay = 0.3
            self.brain.reply("안녕", on_interim=said.append, check_address=True)
        self.assertEqual(said, [])

    def test_wait_lines_go_in_turn_in_three_languages(self):
        for npc, choices in ai.INTERIM_LINES["wait"].items():
            self.assertIn(npc, persona_store.NPC_IDS)
            self.assertGreaterEqual(len(choices), 2)
            for line in choices:
                self.assertEqual(len(line), 3)
                self.assertTrue(all(text and len(text) <= app.WORLD_MAX_LINE for text in line))
            first = [ai.wait_line(npc) for _ in range(len(choices))]
            self.assertEqual(sorted(first), sorted(choices))


class QueueTests(unittest.TestCase):
    def job(self, seq):
        tail = NS(npc_epoch={"kwonjun": 0, "itoa": 0}, npc_state={"kwonjun": True, "itoa": True})
        return NS(tail=tail, npc_epoch=0, vad_end_at=None, seq=seq)

    def test_wait_line_is_dropped_when_the_answer_comes_first(self):
        queue = app.ReplyQueue()
        job = self.job(7)
        queue.put(app.WaitReply({"npc": "kwonjun", "seq": 7, "line": "잠깐만", "more": True}, job))
        queue.put(app.WaitReply({"npc": "itoa", "seq": 8, "line": "잠깐만", "more": True}, self.job(8)))
        queue.put(app.TimedReply({"npc": "kwonjun", "seq": 7, "line": "대답"}, job))
        got = [queue.get_for_world(0.1)["line"], queue.get_for_world(0.1)["line"]]
        self.assertEqual(sorted(got), ["대답", "잠깐만"])            # itoa's own wait line stays
        self.assertTrue(queue.empty())

    def test_wait_line_already_taken_is_not_undone(self):
        queue = app.ReplyQueue()
        job = self.job(9)
        queue.put(app.WaitReply({"npc": "kwonjun", "seq": 9, "line": "잠깐만", "more": True}, job))
        self.assertEqual(queue.get_for_world(0.1)["line"], "잠깐만")
        queue.put(app.TimedReply({"npc": "kwonjun", "seq": 9, "line": "대답"}, job))
        self.assertEqual(queue.get_for_world(0.1)["line"], "대답")


if __name__ == "__main__":
    unittest.main()
