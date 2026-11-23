"""Offline regressions for poll progress, map context, and visitor impressions."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import test_gaze_listen as gaze

import cfg_bridge
import map_knowledge
import persona_ai
import persona_npc_app as app
import persona_store


class Round4Voice(unittest.TestCase):
    def test_waiting_poll_receives_each_stage_before_the_reply(self):
        class Brain(gaze.SlowBrain):
            def __init__(self, store):
                super().__init__(store)
                self.gate.clear()
                self.finish = threading.Event()

            def reply(self, text, on_interim=None, **kw):
                self.heard.append(text)
                self.gate.wait(6)
                on_interim({"interim": "search", "steps": [], "line": "잠깐만!"})
                self.finish.wait(6)
                return {"emotion": "neutral", "intensity": 0.5, "steps": [], "line": "찾았어!",
                        "lines": {"ko": "찾았어!"}, "model": "mock", "ai_ms": 1}

        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "replies", app.ReplyQueue()), \
                patch.object(app, "LOG_PATH", str(Path(tmp) / "voice.tsv")), patch.object(app, "POLL_WAIT_S", 4):
            path, store, _, brain, tr, mic, stop, watcher = gaze.start_watcher(tmp, "poll", "gaze", Brain)
            tr.text = "오늘 날씨 검색해 줘!"
            server = app.make_server(0, store)
            serving = threading.Thread(target=server.serve_forever, daemon=True)
            serving.start()

            def poll():
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=6)
                try:
                    conn.request("GET", "/poll")
                    return json.loads(conn.getresponse().read())
                finally:
                    conn.close()

            def progress(state):
                return {"state": state, "speaker": "Tester One"}

            try:
                gaze.add(path, gaze.LISTEN_ON)
                gaze.wait_for(lambda: mic.tap is not None, "microphone open")
                received = []
                waiting = threading.Thread(target=lambda: received.append(poll()), daemon=True)
                waiting.start()
                gaze.wait_for(lambda: bool(gaze.tsv("HTTP_IN")), "world is waiting on /poll")
                mic.push(gaze.speech(0.65))
                waiting.join(timeout=3)
                self.assertEqual(received, [progress("hearing")])
                self.assertFalse(tr.clips, "hearing was published before transcription")
                mic.push(gaze.room(1.6))
                self.assertEqual(poll(), progress("thinking"))
                brain.gate.set()
                searching = poll()
                act = searching.pop("act")      # T118: the searching status carries the turn's record so far
                self.assertEqual(searching, progress("searching"))
                self.assertEqual([row["k"] for row in act["rows"]][:2], ["heard", "think"])
                brain.finish.set()
                reply = poll()
                self.assertEqual((reply["npc"], reply["line"], reply["steps"]), ("kwonjun", "찾았어!", []))
                self.assertNotIn("state", reply)
            finally:
                brain.gate.set()
                brain.finish.set()
                stop.set()
                watcher.join(timeout=3)
                server.shutdown()
                server.server_close()
                serving.join(timeout=3)

    def test_disconnect_during_headers_or_body_is_quiet(self):
        handler = Mock(close_connection=False)
        for error in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            for writer in (handler.end_headers, handler.wfile.write):
                with self.subTest(error=error, writer=writer):
                    handler.close_connection = False
                    writer.side_effect = error
                    self.assertFalse(app.Handler._send(handler, "{}"))
                    self.assertTrue(handler.close_connection)
                    writer.side_effect = None
        self.assertTrue(app.Handler._send(handler, "{}"))

    def test_owner_label_and_touch_limit_survive_settings_save_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = persona_store.Store(data_dir=tmp, migrate_persona=False)
            store.set_speaker("kwonjun")
            store.after_reply("kwonjun", "안녕", '{"lines":{"ko":"안녕"}}',
                              utterance={"speaker": "kwonjun", "source": "PC 주인 마이크", "time": "", "raw": "안녕"})
            self.assertEqual(store.state()["history"][0]["speaker_label"], "kwonjun(나)")
            self.assertIn("kwonjun(나)", store.prompt_tail("kwonjun"))
            self.assertIn("kwonjun", store.players)
            bridge = cfg_bridge.CfgBridge(store, set(), log=Mock())
            payload = json.loads(bridge.payload())
            self.assertEqual(payload["speaker"], "kwonjun(나)")
            self.assertEqual(payload["players"][0]["name"], payload["speaker"])
            self.assertEqual(bridge.apply({"op": "impression", "player": payload["speaker"], "text": "내 인상"}), "")
            self.assertEqual(store.players["kwonjun"]["note"], "내 인상")
            self.assertNotIn(payload["speaker"], store.players)
            self.assertIn("주인(이 PC에서 음성 앱을 켠 사람)", persona_ai.Brain("fake", store).about_me())
            self.assertTrue(store.add_affection("방문자", "pat"))
            self.assertTrue(store.add_affection("방문자", "hug"))
            store.apply({"players": {"방문자": {"note": "다정함."}}})
            store = persona_store.Store(data_dir=tmp, migrate_persona=False)
            self.assertEqual(store.add_affection("방문자", "pat"), "")
            self.assertEqual(store.add_affection("방문자", "hug"), "")
            self.assertEqual(store.players["방문자"]["note"], "다정함.")
            self.assertTrue(store.for_npc("itoa").add_affection("방문자", "pat"))

    def test_map_prompt_cap_lookup_and_world_move_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = persona_store.Store(data_dir=tmp, migrate_persona=False)
            pos = {"x": 3, "y": 0, "z": 5}
            window = {"name": "창가", "zone": "창 안쪽", "purpose": "도시 보기", "aliases": ["window"],
                      "position": pos, "target": "window", "stand_positions": [{"npc": "kwonjun", "position": pos}]}
            mirror = {"name": "거울", "zone": "창 옆", "purpose": "모습 보기", "position": pos}
            bake = Path(tmp) / "map_knowledge.json"
            bake.write_text(json.dumps({"version": 1, "entries": [window, mirror] + [
                dict(mirror, name=f"물건{i}") for i in range(32)]}), encoding="utf-8")
            with patch.object(map_knowledge, "PATH", bake):
                rows = map_knowledge.load()
                self.assertEqual(len(rows), 30)
                self.assertEqual(len(map_knowledge.select("거실에 뭐 있어?", rows)), 5)
                brain = persona_ai.Brain("fake", store)
                brain.client = object()
                self.assertNotIn("## 맵 안내", brain.system_parts(heard="안녕")[1])
                guide = brain.system_parts(heard="거울이 어디야?")[1]
                self.assertIn("거울 · 창 옆", guide)
                self.assertNotIn("창가 ·", guide)
                self.assertNotIn("move_to", guide.splitlines()[-1])

                result = {"lines": {"ko": "창가로 갈게!"}, "steps": [{"do": "move_to", "target": "window"}]}
                with patch.object(brain, "_ask", return_value=result):
                    self.assertEqual(brain.reply("창가로 가 봐")["steps"][0]["position"], pos)
                    self.assertNotIn("position", brain.reply("창가가 어디야?")["steps"][0])
                with patch.object(brain, "_ask", side_effect=[
                    {"lines": {"ko": "잠깐!"}, "map_lookup": "거울"}, {"lines": {"ko": "창 옆이야!"}}
                ]) as ask:
                    self.assertEqual(brain.reply("아바타 모습은 어디서 봐?")["line"], "창 옆이야!")
                    requests = [call.args[0] for call in ask.call_args_list]
                    self.assertNotIn("## 맵 안내", str(requests[0]["system"]))
                    self.assertIn("거울 · 창 옆", str(requests[1]["system"]))
            self.assertEqual(map_knowledge.load(Path(tmp) / "missing.json"), [])


if __name__ == "__main__":
    unittest.main()
