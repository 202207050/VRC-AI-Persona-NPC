"""Offline regression checks for the R voice-app changes."""
import json
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cfg_bridge
import local_commands
import osc_mute
import persona_ai
import persona_store
import search_links
import smalltalk


class R6VoiceTests(unittest.TestCase):
    def test_loopback_osc_boolean_changes_epoch(self):
        mute = osc_mute.MuteState()
        self.assertIsNone(osc_mute.parse_mute(osc_mute.PREFIX + b",i\0\0\0\0\0\1"))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        stop = threading.Event()
        notes = []
        worker = threading.Thread(target=mute.run, args=(lambda *a, **kw: notes.append(a), stop, port))
        worker.start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                for flag, expected in ((b"T", (True, 1)), (b"F", (False, 2))):
                    deadline = time.monotonic() + 1
                    while mute.snapshot != expected and time.monotonic() < deadline:
                        sender.sendto(osc_mute.PREFIX + b"," + flag + b"\0\0", ("127.0.0.1", port))
                        time.sleep(0.01)
                    self.assertEqual(mute.snapshot, expected)
                self.assertTrue(mute.seen)
                self.assertEqual(notes, [])
        finally:
            stop.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())

    def test_local_commands_do_not_guess_out_of_range_or_indirect_speech(self):
        self.assertEqual(local_commands.world_command("나이트 모드 밝기 30%로."), {"do": "night_mode", "action": "30"})
        for spoken, mode in (("켜", "lq"), ("꺼줘", "off"), ("LQ", "lq"), ("HQ로", "hq"), ("컷아웃", "cutout")):
            self.assertEqual(local_commands.world_command("거울 " + spoken), {"do": "mirror", "action": mode})
        for heard, expected in (
            ("전원한테 거울 켜줘", {"do": "mirror", "action": "lq", "scope": "all"}),
            ("모두 거울 꺼줘", {"do": "mirror", "action": "off", "scope": "all"}),
            ("나한테 거울 켜줘", {"do": "mirror", "action": "lq"}),
            ("야간 모드 30%로 해줘", {"do": "night_mode", "action": "30"}),
            ("다 같이 밤 모드 30%로 해줘", {"do": "night_mode", "action": "30", "scope": "all"}),
        ):
            self.assertEqual(local_commands.world_command(heard), expected)
        for heard in ("나이트 모드 밝기 101%", "나이트 모드 밝기 -1%", "거울 켜도 될까?", "거울 어디 있어?"):
            self.assertIsNone(local_commands.world_command(heard))

    def test_mute_cancels_in_progress_audio_and_resumes_both_listen_modes(self):
        import numpy as np
        import persona_npc_app as app
        from test_gaze_listen import FakeMic, FakeTr, join, LISTEN_ON, LOCAL, PTT, room, speech, wait_for

        for mode in ("ptt", "gaze"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                events = []
                mute = osc_mute.MuteState()
                stop = threading.Event()
                mic, tr = FakeMic(), FakeTr()
                path = Path(tmp) / "output_log_r6.txt"
                path.write_text(join(LOCAL), encoding="utf-8")
                store = persona_store.Store(str(Path(tmp) / "data"), migrate_persona=False)
                worker = threading.Thread(target=app.watch_log, args=(tmp, mic, tr, False),
                                          kwargs={"store": store, "stop": stop, "listen_mode": mode, "mute": mute})

                def append(text):
                    with path.open("a", encoding="utf-8") as file:
                        file.write(text)

                def set_mute(flag):
                    mute.receive(osc_mute.PREFIX + (b",T\0\0" if flag else b",F\0\0"))

                def count(kind):
                    return events.count(kind)

                with patch.object(app, "log", side_effect=lambda kind, *a, **kw: events.append(kind)):
                    worker.start()
                    try:
                        wait_for(lambda: count("WATCH"), "log ready")
                        if mode == "ptt":
                            down, up = PTT.splitlines(keepends=True)
                            append(down)
                            wait_for(lambda: count("PTT_DOWN"), "PTT held")
                            set_mute(True)
                            wait_for(lambda: count("PTT_CANCEL"), "muting cancels held PTT")
                            append(up + PTT)
                            wait_for(lambda: count("PTT_IGNORED"), "muted press ignored")
                        else:
                            append(LISTEN_ON)
                            wait_for(lambda: mic.tap is not None, "listening")
                            mic.push(np.concatenate([room(0.6), speech(0.8)]))
                            wait_for(mic.drained, "partial speech delivered")
                            set_mute(True)
                            wait_for(lambda: mic.tap is None, "muting closes gaze tap")
                            mic.push(np.concatenate([speech(1), room(1.5)]))
                        self.assertEqual(tr.clips, [])
                        set_mute(False)
                        if mode == "ptt":
                            append(PTT)
                        else:
                            wait_for(lambda: mic.tap is not None, "tap reopens after unmute")
                            mic.push(np.concatenate([room(0.6), speech(0.8), room(1.5)]))
                        wait_for(lambda: count("REPLY"), "only fresh speech is answered")
                        self.assertEqual(len(tr.clips), 1)
                    finally:
                        stop.set()
                        worker.join(3)
                        app.clear_replies()

    def test_requested_chat_uses_shared_quota_with_script_fallback(self):
        self.assertTrue(smalltalk.requested("둘이서 대화해봐"))
        self.assertTrue(smalltalk.requested("NPC끼리 얘기해 줘"))
        self.assertFalse(smalltalk.requested("둘이서 대화했어?"))
        with tempfile.TemporaryDirectory() as tmp:
            store = persona_store.Store(tmp, migrate_persona=False)
            store.set_smalltalk_settings({"ai": True, "ratio": 0, "daily_cap": 1})
            chat = smalltalk.SmallTalk(store)
            draft = chat.scripted()
            count = []
            brain = NS(client=True, smalltalk=lambda *a, **kw: count.append(kw) or draft)
            self.assertEqual(chat.reply(brain, lambda: True)[1], "script")
            first, source = chat.reply(brain, lambda: True, requested_by_player=True)
            self.assertEqual(source, "ai")
            self.assertGreaterEqual(len(first["dialog"]), 2)
            self.assertLessEqual(len(first["dialog"]), 4)
            second, source = chat.reply(brain, lambda: True, requested_by_player=True,
                                        opener={"npc": "kwonjun", "line": "이토아, 들어 봐."})
            self.assertEqual(source, "script")
            self.assertEqual([row["npc"] for row in second["dialog"]], ["itoa"])
            self.assertEqual(len(count), 1)
            other = store.for_npc("itoa")
            self.assertTrue(smalltalk.addressed("이토아, 들어 봐.", other))
            self.assertFalse(smalltalk.addressed("이토아가 그러더라.", other))
            chat.remember(second["dialog"][0], {"npc": "kwonjun", "line": "이토아, 들어 봐."})
            self.assertIn("상대 NPC", other.history[-2]["content"])
            self.assertIn(second["dialog"][0]["line"], store.history[-2]["content"])
            self.assertFalse(store.players)
            self.assertFalse(other.players)

    def test_dialog_order_when_first_npc_is_busy_and_epoch_invalidation(self):
        from persona_npc_app import DialogReply, ReplyQueue

        first, second = {"npc": "kwonjun", "line": "하나"}, {"npc": "itoa", "line": "둘"}
        tail = NS(npc_epoch={"kwonjun": 0, "itoa": 0}, npc_state={"kwonjun": True, "itoa": True})
        job = NS(tail=tail, seq=7)
        batch = {"ready": 0, "next": 0}
        replies = ReplyQueue()
        replies.put(DialogReply(first, job, batch, 0, None, True))
        replies.put(DialogReply(second, job, batch, 1, first, False))
        replies.protected_until["kwonjun"] = 0, time.monotonic() + 5
        with self.assertRaises(queue.Empty):
            replies.get_for_world(0)
        del replies.protected_until["kwonjun"]
        self.assertEqual(replies.get_for_world(0)["line"], "하나")
        tail.npc_epoch["itoa"] += 1
        batch["ready"] = 0
        with self.assertRaises(queue.Empty):
            replies.get_for_world(0)

    def test_ptt_dispatches_an_immediate_dialog_and_local_world_command(self):
        import persona_npc_app as app
        from test_gaze_listen import FakeMic, FakeTr, join, LOCAL, PTT, T, wait_for

        with tempfile.TemporaryDirectory() as tmp:
            events = []
            path = Path(tmp) / "output_log_commands.txt"
            path.write_text(join(LOCAL), encoding="utf-8")
            stop, mic, tr = threading.Event(), FakeMic(), FakeTr()
            store = persona_store.Store(str(Path(tmp) / "data"), migrate_persona=False)
            replies = app.ReplyQueue()
            worker = threading.Thread(target=app.watch_log, args=(tmp, mic, tr, False),
                                      kwargs={"store": store, "stop": stop, "listen_mode": "ptt"})

            def append(text):
                with path.open("a", encoding="utf-8") as file:
                    file.write(text)

            with patch.object(app, "log", side_effect=lambda kind, *a, **kw: events.append(kind)), \
                    patch.object(app, "replies", replies):
                worker.start()
                try:
                    wait_for(lambda: "WATCH" in events, "log ready")
                    tr.text = "둘이서 대화해봐"
                    append(T + "[W04AI] NPC_STATE kwonjun=1 itoa=1\n" + PTT)
                    wait_for(lambda: "NPC_CHAT" in events, "explicit chat queued")
                    first = replies.get_for_world(0)
                    self.assertIsInstance(first, app.DialogReply)
                    self.assertTrue(first["more"])
                    first.batch["ready"] = 0
                    self.assertNotEqual(first["npc"], replies.get_for_world(0)["npc"])
                    tr.text = "나이트 모드 밝기 30%로"
                    append(PTT)
                    reply = replies.get_for_world(2)
                    self.assertEqual(reply["steps"], [{"do": "night_mode", "action": "30"}])
                    tr.text = "전원한테 거울 켜줘"
                    append(PTT)
                    reply = replies.get_for_world(2)
                    self.assertEqual(reply["steps"], [{"do": "mirror", "action": "lq"}])
                    self.assertEqual(reply["line"], "거울은 각자 화면에서만 바뀌어서 네 거울만 바꿨어.")
                finally:
                    stop.set()
                    worker.join(3)

    def test_model_lock_and_persistent_shared_choices(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"W04_MODEL_EDITORS": "Owner"}):
            store = persona_store.Store(tmp, migrate_persona=False)
            other = store.for_npc("itoa")
            bridge = cfg_bridge.CfgBridge(store)
            self.assertEqual(store.brain_options()["brain_model"], "claude-haiku-5-5")
            with self.assertRaises(ValueError):
                bridge.change_brain("Visitor", {"brain_model": "claude-opus-5-5"})
            with self.assertRaises(ValueError):
                bridge.change_brain("Visitor", unlocked=True)
            bridge.change_brain("Owner", unlocked=True)
            req = {"npc": "itoa", "op": "model", "model": "claude-opus-5-5", "player": "Owner"}
            self.assertEqual(bridge.apply(req, "log", "Visitor"), "")
            bridge.change_brain("Visitor", {"brain_effort": "high"})
            self.assertEqual(store.brain_options(), other.brain_options())
            self.assertEqual(other.brain_options(), {"brain_model": "claude-opus-5-5", "brain_effort": "high"})
            bridge.world_changed("log")
            self.assertTrue(bridge.apply(req, "log", "Visitor"))
            self.assertEqual(persona_store.Store(tmp, migrate_persona=False).brain_options(), store.brain_options())

    def test_search_result_urls_stay_local(self):
        good, unsafe = "https://www.naver.com/", "https://public.example:bad/"
        blocks = [
            NS(type="web_search_tool_result", content=[
                NS(type="web_search_result", title="네이버 " + good, url=good),
                NS(type="web_search_result", title="bad", url=unsafe)]),
            NS(type="text", text="result"),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"W04_WEB_SEARCH": "1",
                                                                            "W04_LINK_DOMAINS": "naver.com"}):
            store = persona_store.Store(tmp, migrate_persona=False)
            brain = persona_ai.Brain("fake", store)
            brain.client = object()

            def ask(kwargs, total):
                if not kwargs.get("tools"):
                    return {"lines": {"ko": "잠깐만!"}}
                total["links"] = search_links.extract_links(blocks)
                return {"lines": {"ko": f"[네이버]({good}) 열면 돼.", "en": f"Open {good}"}}

            with patch.object(brain, "_ask", side_effect=ask) as mocked:
                result = brain.reply("네이버 링크 보여줘", global_mode=True)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["steps"], [{"do": "show_links", "action": "네이버"}])
            self.assertEqual(result["links"], [{"title": "네이버", "url": good}])
            self.assertNotIn(good, json.dumps(result["lines"], ensure_ascii=False))
            self.assertNotIn(good, store.history[-1]["content"])
            self.assertFalse(search_links.public_https(unsafe))
            self.assertFalse(search_links.public_https("https://localhost/"))


if __name__ == "__main__":
    unittest.main()
