"""Offline coverage for spoken output settings and multilingual /poll replies."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import persona_ai as ai
import persona_npc_app as app
import persona_store
import smalltalk


class ReplyLanguageTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = persona_store.Store(data_dir=folder.name, migrate_persona=False)
        self.brain = ai.Brain("fake", self.store)
        self.brain.client = object()

    def test_spoken_change_is_a_local_command(self):
        with patch.object(self.brain, "_ask") as ask:
            for spoken, language in (
                ("대답을 영어로", "en"), ("권준아, 이제 일본어로 대답해 줘.", "ja"),
                ("한국어로 말해줘", "ko"),
            ):
                with self.subTest(spoken=spoken):
                    reply = self.brain.reply(spoken)
                    self.assertEqual(reply["output_language"], language)
                    self.assertEqual(set(reply["lines"]), {"ko", "en", "ja"})
                    self.assertEqual(app.poll_output(app.make_reply(spoken, 0))["output_language"], language)
            ask.assert_not_called()

        for spoken in ("일본어로 알아들어", "영어로 들어", "영어로 대답해라는 말을 번역해 줘"):
            self.assertIsNone(ai.output_language_reply(spoken))
        self.assertFalse(self.store.history)

    def test_poll_localizes_missed_speech_and_bounds_echo(self):
        missed = app.poll_output(app.make_reply("", 0))
        self.assertEqual(missed["lines"]["en"], "Huh? I didn't catch that.")
        echo = app.poll_output(app.make_reply("가" * 200, 0))
        self.assertEqual(set(echo["lines"]), {"ko", "en", "ja"})
        self.assertTrue(all(len(line) <= app.WORLD_MAX_LINE for line in echo["lines"].values()))
        empty = app.make_empty_reply("note_add", 7)
        self.assertIs(app.poll_output(empty), empty)

    def test_search_interim_and_error_keep_all_languages(self):
        interim = Mock()
        with patch.object(self.brain, "search_on", return_value=True), patch.object(
            self.brain, "_ask", side_effect=[{"search": "서울 날씨", "lines": {"ko": "찾아볼게!"}},
                                             RuntimeError("search stalled")],
        ):
            reply = self.brain.reply("서울 날씨 알려줘", on_interim=interim, global_mode=True)
        first = interim.call_args.args[0]
        self.assertEqual(first["interim"], "search")
        self.assertEqual(first["lines"], ai.localized_lines(ai.INTERIM_LINES["search"]))
        self.assertEqual(reply["lines"], ai.localized_lines(ai.SEARCH_ERROR))

    def test_refusal_and_missing_translation_keep_language_slots(self):
        with patch.object(self.brain, "_ask", side_effect=RuntimeError("refusal")):
            reply = self.brain.reply("다른 이야기", global_mode=True)
        self.assertEqual(reply["lines"], ai.localized_lines(ai.REFUSAL["kwonjun"]))
        with patch.object(self.brain, "_ask", return_value={"lines": {"ko": "영화 볼래?", "en": "Want to watch a movie?"}}):
            reply = self.brain.reply("영화 추천", global_mode=True)
        self.assertEqual(reply["lines"]["en"], "Want to watch a movie?")
        self.assertEqual(reply["lines"]["ja"], ai.UNAVAILABLE_TRANSLATION["ja"])

    def test_translation_log_flags(self):
        for lang, normal in (("en", "Hello!"), ("ja", "こんにちは！")):
            with self.subTest(lang=lang):
                self.assertEqual(app.translation_flag({}, lang), "0")
                self.assertEqual(app.translation_flag({lang: ""}, lang), "0")
                self.assertEqual(app.translation_flag({lang: normal}, lang), "1")
                self.assertEqual(app.translation_flag({lang: ai.UNAVAILABLE_TRANSLATION[lang]}, lang), "fallback")

    def test_incomplete_ai_smalltalk_uses_translated_script(self):
        self.brain.smalltalk = Mock(return_value=[{"npc": "kwonjun", "line": "안녕"}, {"npc": "itoa", "line": "안녕!"}])
        with patch.object(self.store, "reserve_smalltalk_ai", return_value=True):
            reply, source = smalltalk.SmallTalk(self.store).reply(self.brain, lambda: True, global_mode=True)
        self.assertEqual(source, "fallback")
        self.assertTrue(all(set(turn["lines"]) == {"ko", "en", "ja"} for turn in reply["dialog"]))


if __name__ == "__main__":
    unittest.main()
