"""Offline regression tests. Run with pytest or python tests/test_multi_npc.py.

Only temporary data, spare HTTP ports, fake brains/STT/microphones are used.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cfg_bridge
import export_cloud_prompt
import persona_ai
import persona_npc_app as app
import persona_store
import test_gaze_listen as gaze
from test_settings import request


from test_npc_store import StoreFixture


class MultiNpcTests(StoreFixture):
    def setUp(self):
        super().setUp()
        self.log_patch = patch.object(app, "LOG_PATH", str(self.root / "voice.tsv"))
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)

    def test_event_ids_and_legacy_default(self):
        for event in ("LISTEN_ON", "LISTEN_OFF", "PTT_DOWN", "PTT_UP"):
            for suffix, expected in (("", "kwonjun"), (" npc=itoa", "itoa"),
                                     (" npc=kwonjun", "kwonjun"), (" npc=../../bad", "kwonjun")):
                line = f"[W04AI] {event}{suffix} held_ms=1000"
                self.assertEqual(app.EVENT_RE.search(line).group(1), event)
                self.assertEqual(app.event_npc(line), expected)

    def test_state_backlog_without_replaying_listen(self):
        path = self.root / "output_log_test.txt"
        path.write_text(gaze.join(gaze.LOCAL) + gaze.T + "[W04AI] NPC_STATE kwonjun=1 itoa=0\n"
                        + gaze.LISTEN_ON, encoding="utf-8")
        tail = app.LogTail(str(path))
        try:
            self.assertEqual(tail.npc_state, {"kwonjun": True, "itoa": False})
            self.assertEqual(list(tail.lines()), [])
        finally:
            tail.close()



    def test_brains_lazy_and_fixed_prompts_independent(self):
        with patch.object(app, "make_brain", side_effect=lambda mode, st: persona_ai.Brain("fake", st)) as factory:
            brains = app.NpcBrains("fake", self.store)
            self.assertEqual(factory.call_count, 0)
            first = brains.for_npc("itoa")
            self.assertIs(first, brains.for_npc("itoa"))
            second = brains.for_npc()
            self.assertEqual(factory.call_count, 2)
            self.assertIn("귤", first.system_blocks()[0]["text"])
            self.assertNotEqual(first.system_blocks()[0]["text"], second.system_blocks()[0]["text"])
            self.assertEqual(first.system_blocks()[0]["cache_control"], {"type": "ephemeral"})
            first.reply("안녕")
            self.assertEqual(len(self.itoa.history), 2)
            self.assertEqual(self.store.history, [])
            self.itoa.set_npc_name("새 이름")
            self.assertIn("새 이름", first.system_blocks()[0]["text"])
            self.assertEqual(self.store.npc_name(), "kwonjun")




    def test_http_settings_save_load_and_poll(self):
        server = app.make_server(0, self.store, bridge=self.bridge, token="test-token")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        headers = {"X-W04-Token": "test-token"}
        try:
            status, html = request(port, "GET", "/settings?t=test-token")
            self.assertEqual(status, 200)
            self.assertIn('id="npcSelect"', html)
            status, text = request(port, "POST", "/api/save", {"npc": "itoa", "notes": ["only itoa"]}, headers)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(text)["npc_name"], "ITOAR")
            self.assertEqual(json.loads(text)["notes"][0]["text"], "only itoa")
            for suffix, expected in (("", []), ("?npc=itoa", ["only itoa"])):
                status, text = request(port, "GET", "/api/state" + suffix, headers=headers)
                self.assertEqual(status, 200)
                self.assertEqual([n["text"] for n in json.loads(text)["notes"]], expected)
            self.assertEqual(request(port, "GET", "/api/state?npc=itoa")[0], 403)
            while not app.replies.empty():
                app.replies.get_nowait()
            app.replies.put({"line": "귤 먹자", "npc": "itoa"})
            self.assertEqual(json.loads(request(port, "GET", "/poll")[1])["npc"], "itoa")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def _watch(self, mode):
        folder = self.root / "logs"
        folder.mkdir()
        path = folder / "output_log_test.txt"
        path.write_text(gaze.join(gaze.LOCAL), encoding="utf-8")
        mic, tr = gaze.FakeMic(), gaze.FakeTr()
        tr.text = "안녕"
        brains = app.NpcBrains("fake", self.store)
        stop = threading.Event()
        watcher = threading.Thread(target=app.watch_log,
                                   args=(str(folder), mic, tr, False, brains, self.store, self.bridge, stop),
                                   kwargs={"listen_mode": mode}, daemon=True)
        while not app.replies.empty():
            app.replies.get_nowait()
        watcher.start()
        gaze.wait_for(lambda: Path(app.LOG_PATH).exists() and "WATCH" in Path(app.LOG_PATH).read_text(encoding="utf-8"), "watching")
        self.addCleanup(lambda: (stop.set(), watcher.join(3)))
        return str(path), mic, tr, brains

    def test_ptt_routes_at_press_and_respects_off_state(self):
        path, mic, tr, brains = self._watch("ptt")
        self.store.apply({"settings": {"stt_hints": "404 -> 권준"}})
        self.itoa.apply({"settings": {"stt_hints": "404 -> 사공사"}})
        tr.text = "404님"
        gaze.add(path, gaze.T + "[W04AI] NPC_STATE kwonjun=1 itoa=1\n"
                 + gaze.T + "[W04AI] PTT_DOWN npc=itoa\n"
                 + gaze.T + "[W04AI] PTT_UP npc=kwonjun held_ms=1500\n")
        reply = app.replies.get(timeout=5)
        self.assertEqual((reply["npc"], reply["heard"]), ("itoa", "사공사님"))
        self.assertEqual(self.itoa.history[0]["content"], "사공사님")
        self.assertEqual(len(self.itoa.history), 2)
        self.assertEqual(self.store.history, [])
        gaze.add(path, gaze.T + "[W04AI] NPC_STATE kwonjun=1 itoa=0\n"
                 + gaze.T + "[W04AI] PTT_DOWN npc=itoa\n"
                 + gaze.T + "[W04AI] PTT_UP npc=itoa held_ms=1500\n")
        self.assertTrue(gaze.no_reply())
        gaze.add(path, gaze.PTT)
        reply = app.replies.get(timeout=5)
        self.assertEqual((reply["npc"], reply["heard"]), ("kwonjun", "권준님"))

    def test_global_log_is_scoped_to_the_current_allowed_world(self):
        path, _, _, brains = self._watch("ptt")
        brain = brains.for_npc("kwonjun")
        with patch.object(brain, "reply", wraps=brain.reply) as answer:
            for marker, expected in (
                (gaze.T + "[W04AI] GLOBAL on=1\n", True),
                (gaze.T + "[W04AI] GLOBAL on=0\n", False),
                (gaze.join("wrld_1234") + gaze.T + "[W04AI] GLOBAL on=1\n" + gaze.join(gaze.LOCAL), False),
                (gaze.T + "[W04AI] GLOBAL on=1\n" + gaze.join("local:different"), False),
                (gaze.T + "[W04AI] GLOBAL on=1\n" + gaze.join("local:different"), False),
            ):
                gaze.add(path, marker + gaze.PTT)
                app.replies.get(timeout=5)
                self.assertEqual(answer.call_args.kwargs.get("global_mode", False), expected)

    def test_gaze_routes_at_onset_and_ignores_other_npc_off(self):
        path, mic, tr, brains = self._watch("gaze")
        gaze.add(path, gaze.T + "[W04AI] NPC_STATE kwonjun=1 itoa=1\n"
                 + gaze.T + "[W04AI] LISTEN_ON npc=itoa\n")
        gaze.wait_for(lambda: mic.tap is not None, "listening")
        mic.push(gaze.speech(1.0))
        gaze.wait_for(mic.drained, "first half")
        gaze.add(path, gaze.T + "[W04AI] LISTEN_ON npc=kwonjun\n"
                 + gaze.T + "[W04AI] LISTEN_OFF npc=itoa\n")
        gaze.wait_for(lambda: Path(app.LOG_PATH).read_text(encoding="utf-8").count("\tLISTEN_ON\t") >= 2, "switched")
        mic.push(gaze.room(1.4))
        self.assertEqual(app.replies.get(timeout=5)["npc"], "itoa")
        self.assertIsNotNone(mic.tap)
        tr.text = "다음 대화"
        gaze.wait_for(lambda: len(self.itoa.history) == 2, "finished")
        time.sleep(0.15)  # allow the worker to release the one-turn gate
        mic.push(gaze.speech(1.0))
        mic.push(gaze.room(1.4))
        self.assertEqual(app.replies.get(timeout=5)["npc"], "kwonjun")



if __name__ == "__main__":
    unittest.main()
