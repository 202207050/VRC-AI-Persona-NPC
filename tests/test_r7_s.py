"""Offline checks for r7-S reply latency and thread recovery."""

import builtins
import io
import os
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_gaze_listen as gaze

app = gaze.app


class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.queue = app.ReplyQueue()
        self.tail = type("Tail", (), {})()
        self.tail.npc_epoch = {"kwonjun": 0, "itoa": 0}
        self.tail.npc_state = {"kwonjun": True, "itoa": True}
        self.job = SimpleNamespace(tail=self.tail, npc_epoch=0)
        self.clock = Mock(return_value=100)
        clock_patch = patch.object(app.time, "monotonic", self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def status(self, state, npc="kwonjun"):
        self.queue.status(state, self.tail, npc, 0, "tester")

    def poll(self):
        return self.queue.get_for_world(0, include_state=True)

    def test_ready_reply_wins_and_clears_matching_state(self):
        self.status("hearing")
        self.status("hearing", "itoa")
        reply = app.TimedReply({"npc": "kwonjun", "line": "응"}, self.job, final=False)
        self.queue.put(reply)
        self.assertIs(self.poll(), reply)
        self.assertEqual(self.poll(), {"state": "hearing", "speaker": "tester"})
        with self.assertRaises(queue.Empty):
            self.poll()

    def test_fast_answer_replaces_deferred_thinking_at_enqueue(self):
        self.status("thinking")
        self.clock.return_value += 1.8
        with self.assertRaises(queue.Empty):
            self.poll()
        reply = app.TimedReply({"npc": "kwonjun", "line": "응"}, self.job)
        self.queue.put(reply)
        self.assertFalse(self.queue.statuses)
        self.assertIs(self.poll(), reply)

    def test_overdue_thinking_can_go_out(self):
        self.status("thinking")
        self.clock.return_value += 3.01
        self.assertEqual(self.poll()["state"], "thinking")

    def test_poll_clock_and_clamped_median_of_last_twenty_ai_times(self):
        self.assertEqual(self.queue.typical_ai_s, 1.8)
        self.queue.poll_served()
        self.assertEqual(self.queue.last_served, 100)
        self.queue.record_ai_time(0)
        self.assertEqual(self.queue.typical_ai_s, 1.0)
        for _ in range(20):
            self.queue.record_ai_time(9000)
        self.assertEqual(self.queue.typical_ai_s, 4.0)
        self.assertEqual(len(self.queue.ai_times), 20)
        self.clock.return_value += app.POLL_MIN_INTERVAL_S
        self.status("thinking")
        self.assertEqual(self.poll()["state"], "thinking")

    def test_interim_enqueue_keeps_state_and_stale_generation_is_pruned(self):
        self.status("thinking")
        interim = app.SearchReply({"npc": "kwonjun"}, self.job)
        self.queue.put(interim)
        self.assertTrue(self.queue.statuses)
        self.tail.npc_epoch["kwonjun"] += 1
        with self.assertRaises(queue.Empty):
            self.poll()
        self.assertFalse(self.queue.statuses)

    def test_dictation_reply_uses_its_own_sequence(self):
        with patch.object(app, "_seq", 99):
            self.assertEqual(app.make_empty_reply("note_add", 7)["seq"], 7)


class VoiceTests(unittest.TestCase):
    def test_fragments(self):
        for text in ("앉아 봐", "TV 켜 줘", "이리 와", "응", "응!", "그만해", "원래대로 돌려놔", "가", "맞으니까"):
            with self.subTest(text=text):
                self.assertFalse(app.is_fragment(text))
        for text in ("TV 켜고", "근데", "그리고 저기", "소파에 가서", "내가 어제는", "어", "음", "네가"):
            with self.subTest(text=text):
                self.assertTrue(app.is_fragment(text))

    def test_startup_builds_both_brains_once_without_sending_requests(self):
        store = Mock()
        store.for_npc.side_effect = lambda npc: npc
        brain = app.NpcBrains("haiku", store)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "unit-test"}), \
                patch.object(app.persona_ai, "Brain", side_effect=lambda mode, npc: Mock()) as constructor, \
                patch.object(app, "log") as log:
            warm = app.start_brain_warmup(brain)
            warm.join(2)
            self.assertFalse(warm.is_alive())
            self.assertEqual(set(brain.brains), {"kwonjun", "itoa"})
            self.assertEqual(constructor.call_count, 2)
            for npc in ("kwonjun", "itoa"):
                brain.brains[npc].reply.assert_not_called()
                brain.for_npc(npc).reply("안녕")
                self.assertEqual(brain.take_init_ms(npc), 0)
            self.assertEqual(constructor.call_count, 2)
            self.assertEqual([call.args[0] for call in log.call_args_list], ["BRAIN_READY", "BRAIN_READY"])
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            self.assertIsNone(app.start_brain_warmup(app.NpcBrains("haiku", store)))

    def test_failed_warmup_can_retry(self):
        brain = app.NpcBrains("haiku", Mock())
        ready = Mock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "unit-test"}), \
                patch.object(app.persona_ai, "Brain", side_effect=[RuntimeError("private"), ready, ready]), \
                patch.object(app, "log"), patch.object(app, "say_line") as console:
            app.start_brain_warmup(brain).join(2)
            self.assertNotIn("kwonjun", brain.brains)
            self.assertIs(brain.for_npc("kwonjun"), ready)
            self.assertNotIn("private", str(console.call_args))

    def test_thread_restarts_after_one_exception(self):
        def fail_once():
            if run.call_count == 1:
                raise OSError("private transcript")

        run = Mock(side_effect=fail_once)
        with patch.object(app.time, "sleep") as sleep, patch.object(app, "log") as log:
            app.supervise_thread("듣기", run)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertRegex(log.call_args.args[1], r"^name=듣기\tOSError\ttest_r7_s\.py:\d+$")

    def test_thread_stops_after_five_failures(self):
        def fail():
            raise RuntimeError("private transcript")

        run = Mock(side_effect=fail)
        with patch.object(app.time, "sleep") as sleep, patch.object(app, "log") as log:
            app.supervise_thread("감시", run)
        self.assertEqual(run.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertIn("앱을 다시 켜 주세요", log.call_args.kwargs["say"])

    def test_gaze_continues_when_log_file_is_locked(self):
        with TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "voice_log.tsv")
            locked = True
            console = io.StringIO()

            def open_log(path, *args, **kwargs):
                if path == log_path and locked:
                    raise PermissionError("locked")
                return builtins.open(path, *args, **kwargs)

            with patch.object(app, "LOG_PATH", log_path), patch.object(app, "_log_missed", 0), \
                    patch.object(app, "replies", app.ReplyQueue()), patch.object(app, "open", open_log, create=True), \
                    patch("sys.stdout", console):
                path, _, _, brain, tr, mic, stop, watcher = gaze.start_watcher(tmp, "s_log", "gaze")
                try:
                    gaze.add(path, gaze.LISTEN_ON)
                    gaze.wait_for(lambda: mic.tap is not None, "gaze tap opened")
                    tr.text = "앉아 봐"
                    mic.push(gaze.speech(0.8))
                    mic.push(gaze.room(1.4))
                    self.assertEqual(app.replies.get(timeout=5)["heard"], "앉아 봐")
                    self.assertFalse(app.replies.statuses)
                    locked = False
                    brain.gate.clear()
                    tr.text = "이리 와"
                    mic.push(gaze.speech(0.8))
                    mic.push(gaze.room(1.4))
                    gaze.wait_for(lambda: len(brain.heard) == 2, "second utterance reached the worker")
                    mic.push(gaze.speech(0.8))
                    mic.push(gaze.room(1.4))
                    gaze.wait_for(lambda: bool(gaze.tsv("SPEECH_DROPPED")), "busy utterance was logged")
                    brain.gate.set()
                    self.assertEqual(app.replies.get(timeout=5)["heard"], "이리 와")
                    self.assertEqual(len(tr.clips), 2)
                finally:
                    brain.gate.set()
                    stop.set()
                    watcher.join(2)
            self.assertEqual(console.getvalue().count("기록 파일을 쓸 수 없어요"), 1)
            log = Path(log_path).read_text(encoding="utf-8")
            self.assertRegex(log, r"\tLOG_RECOVERED\tmissed=[1-9]\d*")
            self.assertRegex(log, r"\tSPEECH_DROPPED\tnpc=kwonjun\treason=busy\tspeech_s=0\.\d+")


if __name__ == "__main__":
    unittest.main()
