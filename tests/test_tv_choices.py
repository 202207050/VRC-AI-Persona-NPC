"""Offline checks for TV history and a questioner's world choice."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import http.client
import json
import subprocess
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cfg_bridge
import persona_ai
import persona_npc_app as app
import persona_store
import tv_history as tv
from choice_relay import ChoiceMailbox, read_event

VIDEO = "dQw4w9WgXcQ"
URL = "https://youtu.be/" + VIDEO
STAMP = "2026.11.20 20:00:00 Debug      -  "


def played(url, title):
    return "[W04AI] TV_HISTORY " + json.dumps({"url": url, "title": title, "key": tv.world_key(url)})


class TvChoiceTests(unittest.TestCase):
    def test_raw_log_and_confirmed_history(self):
        playback = tv.PlaybackLog()
        self.assertEqual(tv.world_key(URL), "h8fd69810")
        self.assertEqual(playback.feed(STAMP + f"[Video Playback] Attempting to resolve URL '{URL}'\n"), VIDEO)
        self.assertEqual(list(playback.rows), [(URL, "", "")])
        self.assertFalse(playback.active)
        for i in range(11):
            playback.feed(played(f"https://example.invalid/watch/{i}", f"영상 {i}"))
        self.assertEqual(len(playback.rows), 10)
        self.assertEqual(playback.rows[-1][1], "영상 1")
        playback.feed(played("https://example.invalid/watch/3", "찾은 제목"))
        self.assertEqual(playback.rows[0][1], "찾은 제목")
        self.assertEqual(len(playback.rows), 10)
        playback.feed(STAMP + f"[Video Playback] Attempting to resolve URL '{URL}'\n")
        self.assertEqual(playback.rows[0][1], "찾은 제목")
        playback.feed("[W04AI] TV_HISTORY_IDLE")
        self.assertFalse(playback.active)
        self.assertEqual(tv.youtube_id("https://youtube.com.invalid/watch?v=" + VIDEO), "")
        self.assertEqual(tv.youtube_id("https://youtube.com/shorts/" + VIDEO + "?x=1"), VIDEO)

    def test_web_requests_are_metadata_only(self):
        page = Mock()
        page.__enter__ = Mock(return_value=page)
        page.__exit__ = Mock()
        page.read.return_value = b'{"title":"a title"}'
        with patch.object(tv, "urlopen", return_value=page) as open_url:
            self.assertEqual(tv.oembed(URL), "a title")
        request = open_url.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://www.youtube.com/oembed?"))
        self.assertEqual(request.get_header("User-agent"), tv.UA)
        self.assertEqual(open_url.call_args.kwargs["timeout"], 4)

        def run(args, **kwargs):
            for flag in ("--ignore-config", "--skip-download", "--write-auto-subs",
                         "--dump-single-json", "--no-playlist"):
                self.assertIn(flag, args)
            self.assertEqual(args[-2:], ["--", URL])
            self.assertEqual(kwargs["timeout"], 14)
            Path(kwargs["cwd"], "caption.ko.json3").write_text(
                '{"events":[{"segs":[{"utf8":"captions"}]}]}', encoding="utf-8")
            return SimpleNamespace(stdout='{"title":"video", "description":"desc"}')

        with TemporaryDirectory() as root:
            exe = Path(root, "yt-dlp.exe")
            exe.touch()
            with patch.object(tv.subprocess, "run", side_effect=run):
                self.assertEqual(tv.metadata(exe, URL), ("video", "desc", "captions"))
            with patch.object(tv.subprocess, "run", side_effect=subprocess.TimeoutExpired("yt-dlp", 14)):
                self.assertEqual(tv.metadata(exe, URL), ("", "", ""))

    def test_cache_is_shared_by_video_and_prompt_is_conditional(self):
        with TemporaryDirectory() as root:
            summarize = Mock(return_value="첫 줄\n둘째 줄")
            with patch.object(tv, "oembed", return_value="제목") as oembed, patch.object(
                tv, "metadata", return_value=("", "설명", "자동자막")
            ):
                cache = tv.VideoCache(root, Path(root, "yt-dlp.exe"), summarize)
                cache.watch(VIDEO)
                cache.watch(VIDEO)
                cache.pool.shutdown()
                again = tv.VideoCache(root, Path(root, "yt-dlp.exe"), summarize)
                again.watch(VIDEO)
                again.pool.shutdown()
            oembed.assert_called_once()
            summarize.assert_called_once_with("제목", "설명", "자동자막")
            playback = tv.PlaybackLog()
            playback.feed(played(URL, "기존 제목"))
            self.assertEqual(again.context(playback, "지금 장 보고 왔어"), [])
            self.assertEqual(again.context(playback, "영상 통화 하자"), [])
            rows = again.context(playback, "아까 그 영상 다시 틀어 줘")
            self.assertEqual(rows, [{"title": "제목", "key": "h8fd69810", "summary": "첫 줄\n둘째 줄", "state": "current"}])
            self.assertEqual(again.context(playback, "두 번째", "어떤 영상?"), rows)

    def test_model_choices_and_replay_are_validated(self):
        data = {"lines": {"ko": "어떤 영상?"}, "choices": ["<바다>", "우주", "기타", "네 번째"],
                "steps": [{"do": "tv_replay", "action": "h8fd69810"}]}
        got = persona_ai._clamp_reply(data)
        self.assertEqual(got["choices"], ["바다", "우주"])
        self.assertEqual(got["steps"], [])
        self.assertEqual(persona_ai.clamp_steps([{"do": "tv_replay", "action": URL}]), [])
        with TemporaryDirectory() as root:
            brain = persona_ai.Brain("fake", persona_store.Store(data_dir=root, migrate_persona=False))
            brain.client = object()
            data.pop("choices")
            rows = [{"title": "바다", "key": "h8fd69810", "summary": "요약"}]
            with patch.object(brain, "_ask", return_value=data) as ask:
                self.assertEqual(brain.reply("안녕")["steps"], [])
                self.assertNotIn("TV 지금과 최근 재생 기록", ask.call_args.args[0]["messages"][-1]["content"])
                reply = brain.reply("이 영상 다시 틀어 줘", tv_rows=rows,
                                    utterance={"raw": "이 영상 다시 틀어 줘"})
                self.assertEqual(reply["steps"][0]["action"], "h8fd69810")
                message = ask.call_args.args[0]["messages"][-1]["content"]
                self.assertIn("요약", message)
                self.assertIn("[발화 기록 데이터]", message)

    def test_world_button_to_local_reply(self):
        with TemporaryDirectory() as root:
            store = persona_store.Store(data_dir=root, migrate_persona=False)
            brain = persona_ai.Brain("fake", store)
            got = {"model": "fake", "ai_ms": 0, "emotion": "neutral", "intensity": .5, "steps": [],
                   "lines": {"ko": "그래, 바다 영상으로 할게."}, "line": "그래, 바다 영상으로 할게."}
            brain.reply = Mock(return_value=got)
            bridge = cfg_bridge.CfgBridge(store, set())
            log_path = Path(root, "output_log_2026-11-20_20-00-00.txt")
            log_path.write_text(STAMP + "[Behaviour] Joining local:abcdef123\n", encoding="utf-8")
            marker = "[W04AI] CHOICE_INPUT "
            data = {"npc": "kwonjun", "id": "120:1:4", "text": "바다", "context": ["영상 다시 틀어", "어떤 영상?"]}
            self.assertIsNone(read_event(marker + json.dumps(dict(data, id="wrong")), marker))
            stop = threading.Event()
            mailbox = ChoiceMailbox()
            ready = threading.Event()
            original = app.LogTail

            def open_tail(path):
                tail = original(path)
                ready.set()
                return tail

            with patch.object(app, "choice_mailbox", mailbox), patch.object(app, "LogTail", side_effect=open_tail), patch.object(app, "log"):
                server = app.make_server(0, store, brain, bridge=bridge)
                serving = threading.Thread(target=server.serve_forever, daemon=True)
                serving.start()
                thread = threading.Thread(target=app.watch_log, args=(root, None, None, False, brain, store, bridge, stop),
                                          kwargs={"listen_mode": "ptt"}, daemon=True)
                thread.start()
                try:
                    self.assertTrue(ready.wait(2))
                    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=4)
                    self.addCleanup(conn.close)
                    conn.request("GET", "/choice/kwonjun", headers={"Origin": "https://example.invalid"})
                    denied = conn.getresponse()
                    self.assertEqual(denied.status, 403)
                    denied.read()
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(STAMP + marker + json.dumps(data, ensure_ascii=False) + "\n")
                    conn.request("GET", "/choice/kwonjun")
                    result = conn.getresponse()
                    self.assertEqual(result.status, 200)
                    reply = json.loads(result.read())
                    self.assertEqual(reply["choice_id"], data["id"])
                    self.assertEqual(reply["heard"], "바다")
                    self.assertEqual(brain.reply.call_args.args, ("바다",))
                    self.assertEqual(brain.reply.call_args.kwargs["clarification"], data["context"])
                    self.assertEqual(mailbox.take("itoa", 0), {})
                    self.assertEqual(mailbox.take("kwonjun", 0), {})
                finally:
                    stop.set()
                    thread.join(2)
                    server.shutdown()
                    server.server_close()
                    serving.join(2)


if __name__ == "__main__":
    unittest.main()
