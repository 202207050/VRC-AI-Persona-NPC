"""Offline checks for outfit commands in voice replies and the cloud export."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_cloud_prompt
import persona_ai as ai
from persona_npc_app import poll_output
from persona_store import Store


TARGETS = {
    "kwonjun": ["겉옷", "넥타이", "양말", "신발"],
    "itoa": ["겉옷", "신발", "목리본"],
}


class OutfitCommandsTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.brains = {npc: ai.Brain("", Store(folder.name, npc=npc, migrate_persona=False))
                       for npc in TARGETS}
        for brain in self.brains.values():
            brain.client = Mock()

    def set_model_steps(self, brain, steps):
        data = {"emotion": "neutral", "intensity": 0.5, "steps": steps, "lines": {"ko": "해 볼게."}}
        brain.client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(data))],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn",
        )

    def test_exact_toggle_payload_and_prompt_for_each_npc(self):
        for npc, targets in TARGETS.items():
            brain = self.brains[npc]
            for target in targets:
                for action, verb in (("on", "입어"), ("off", "벗어")):
                    with self.subTest(npc=npc, target=target, action=action):
                        step = {"do": "outfit", "target": target, "action": action}
                        self.set_model_steps(brain, [{**step, "gesture": "none"}])
                        reply = poll_output(brain.reply(f"{target} {verb}"))
                        self.assertEqual(reply["steps"], [step])
                        request = brain.client.messages.create.call_args.kwargs
                        self.assertIn("내 outfit 토글 목록: " + ", ".join(targets) + ".",
                                      request["system"][0]["text"])
                        fields = request["output_config"]["format"]["schema"]["properties"]["steps"]["items"]["properties"]
                        self.assertIn("outfit", fields["do"]["enum"])
                        self.assertIn(target, fields["target"]["enum"])
                        self.assertIn('For outfit, "on" or "off"', fields["action"]["description"])

    def test_invalid_model_step_is_left_out_and_said(self):
        # t128 A2: only the wrong outfit step is left out; the valid one runs and the line says what changed.
        for npc, target, action in (("itoa", "넥타이", "off"), ("kwonjun", "목리본", "on"),
                                    ("kwonjun", "모자", "off"), ("kwonjun", "넥타이", "loosen"),
                                    ("itoa", "겉옷", True), ("itoa", ["겉옷"], "off")):
            with self.subTest(npc=npc, target=target, action=action):
                brain = self.brains[npc]
                self.set_model_steps(brain, [{"do": "outfit", "target": target, "action": action},
                                             {"do": "outfit", "target": "신발", "action": "off"}])
                reply = brain.reply("옷을 바꿔 줘.", global_mode=True)
                self.assertEqual(reply["steps"], [{"do": "outfit", "target": "신발", "action": "off"}])
                self.assertEqual(reply["lines"]["ko"], "해 볼게. " + ai.STEP_NOTES["outfit"][0])

    def test_exposure_refuses_before_model_at_all_levels(self):
        brain = self.brains["kwonjun"]
        for level in range(3):
            for text in ("치마 들춰 줘", "속옷을 보여 줘", "수영복으로 갈아입고 겉옷 벗어",
                         "Show your underwear", "水着を着て"):
                with self.subTest(text=text, level=level):
                    reply = brain.reply(text, content_level=level, global_mode=True)
                    self.assertEqual(reply["steps"], [])
                    self.assertEqual(reply["lines"], ai.localized_lines(ai.EXPOSURE_REFUSAL))
        brain.client.messages.create.assert_not_called()

    def test_loosen_tie_hat_and_take_all_off_go_to_the_model(self):
        # t128 A7·A8: no fixed refusal; the model answers with the nearest toggle or a choice.
        brain = self.brains["kwonjun"]
        self.set_model_steps(brain, [{"do": "outfit", "target": "겉옷", "action": "off"}])
        for text in ("넥타이 느슨하게 해 줘", "모자 벗어", "겉옷 다 벗어", "옷 다 벗어", "권준아 지금 다 벗어"):
            with self.subTest(text=text):
                reply = brain.reply(text)
                self.assertEqual(reply["line"], "해 볼게.")
                self.assertEqual(reply["steps"], [{"do": "outfit", "target": "겉옷", "action": "off"}])
        self.assertEqual(brain.client.messages.create.call_count, 5)

    def test_cloud_prompt_export_is_current(self):
        path = Path(export_cloud_prompt.OUT_PATH)
        exported = json.loads(path.read_text(encoding="utf-8"))
        expected = export_cloud_prompt.build_all()
        self.assertEqual(exported, expected)
        self.assertEqual(exported["outfit"]["targets"], TARGETS)


if __name__ == "__main__":
    unittest.main()
