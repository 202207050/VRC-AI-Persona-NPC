"""Offline checks for the cached voice NPC self-description."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import persona_ai
import persona_store


class SelfDescriptionTests(unittest.TestCase):
    def test_capabilities_are_in_the_cached_block_for_both_npcs(self):
        with tempfile.TemporaryDirectory() as data_dir:
            for npc, name in persona_store.DEFAULT_NPC_NAMES.items():
                with self.subTest(npc=npc):
                    store = persona_store.Store(data_dir=data_dir, npc=npc, migrate_persona=False)
                    brain = persona_ai.Brain("fake", store)
                    brain.client = object()  # build a live-mode prompt without calling the API
                    for enabled in ("1", "0"):
                        with patch.dict("os.environ", {"W04_WEB_SEARCH": enabled}):
                            blocks = brain.system_blocks()
                            description = brain.about_me()
                            self.assertIn(description, blocks[0]["text"])
                            self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
                            self.assertTrue(all(description not in block["text"] for block in blocks[1:]))
                            self.assertIn(f"내 이름은 {name}.", description)
                            self.assertIn("약 6m", description)
                            self.assertIn("TV 영상 재생", description)
                            self.assertIn(f"최근 대화 {persona_store.MAX_HISTORY}번", description)
                            codes = persona_ai.ACTIONS + persona_ai.TARGETS + persona_ai.GESTURES + persona_ai.EMOTIONS
                            for code in set(codes) - {"none"}:
                                self.assertIn(persona_ai.SELF_LABELS[code], description)
                            if enabled == "1":
                                self.assertIn(f"검색 요청당 최대 {persona_ai.SEARCH_MAX_USES}회", description)
                            else:
                                self.assertIn("지금은 웹 검색을 할 수 없다", description)
                            self.assertIsNone(persona_ai._MODEL_NAME.search(blocks[0]["text"]))
                            self.assertIn(persona_ai.MODEL_SECRET_LINE, blocks[0]["text"])

    def test_output_filter_keeps_model_and_provider_private(self):
        for line in ("나는 Claude Haiku야.", "앤트로픽이 만들었어."):
            with self.subTest(line=line):
                reply = persona_ai._clamp_reply({"lines": {"ko": line}})
                self.assertEqual(reply["line"], persona_ai.MODEL_SECRET_LINE)
                self.assertEqual(reply["lines"]["ko"], persona_ai.MODEL_SECRET_LINE)

    def test_global_uses_one_call_and_keeps_the_fixed_prompt_and_history_korean_only(self):
        with tempfile.TemporaryDirectory() as data_dir:
            store = persona_store.Store(data_dir=data_dir, migrate_persona=False)
            brain = persona_ai.Brain("fake", store)
            brain.client = object()
            data = {"lines": {"ko": "영화 볼래?", "en": "  Tickets\n for 12500? ", "ja": "アンソロピックのモデルだよ。"}}
            with patch.object(brain, "_ask", side_effect=lambda *_: json.loads(json.dumps(data))) as ask:
                on = brain.reply("안녕", global_mode=True)
                self.assertEqual(ask.call_count, 1)
                on_request = ask.call_args.args[0]
                off = brain.reply("안녕")
                self.assertEqual(ask.call_count, 2)
                off_request = ask.call_args.args[0]
            self.assertEqual(on_request["system"][0], off_request["system"][0])
            self.assertEqual(on_request["system"][-1], {"type": "text", "text": persona_ai.GLOBAL_ORDER})
            self.assertNotIn(persona_ai.GLOBAL_ORDER, str(off_request["system"]))
            # Global ON makes en/ja required (Haiku skipped the optional fields live); OFF keeps ko only.
            self.assertEqual(off_request["output_config"]["format"]["schema"]["properties"]["lines"]["required"], ["ko"])
            lines = on_request["output_config"]["format"]["schema"]["properties"]["lines"]
            self.assertEqual(lines["required"], ["ko", "en", "ja"])
            self.assertEqual(set(lines["properties"]), {"ko", "en", "ja"})
            self.assertEqual(lines["required"], ["ko", "en", "ja"])
            self.assertEqual(on["lines"], {"ko": "영화 볼래?", "en": "Tickets for 12,500?", "ja": "それは秘密だよ！"})
            self.assertEqual(off["lines"], {"ko": "영화 볼래?"})
            self.assertEqual(json.loads(off_request["messages"][-2]["content"])["lines"], off["lines"])
            long = persona_ai._clamp_reply({"lines": {"ko": "좋아", "en": "x" * 101, "ja": "2026年に12500円"}}, True)
            self.assertEqual(long["lines"], {"ko": "좋아", "en": "x" * 80 + "…", "ja": "2026年に12,500円"})


if __name__ == "__main__":
    unittest.main()
