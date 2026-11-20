"""Offline checks for the distance and prone replies sent to the world."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import persona_ai as ai
from persona_npc_app import poll_output
from persona_store import Store
from smalltalk import clamp_dialog


class WorldMotionCommandsTest(unittest.TestCase):
    def test_voice_commands_reach_poll_payload(self):
        cases = (
            ("2m로 와봐", {"do": "keep_distance", "action": "200"}),
            ("1미터 떨어져", {"do": "keep_distance", "action": "100"}),
            ("더 가까이 와", {"do": "keep_distance", "action": "60"}),
            ("12m 떨어져", {"do": "keep_distance", "action": "1200"}),
            ("소파에 엎드려", {"do": "prone"}),
            ("엎드려서 TV 봐", {"do": "prone_tv"}),
        )
        with tempfile.TemporaryDirectory() as folder:
            brain = ai.Brain("fake", Store(folder, migrate_persona=False))
            brain.client = Mock()
            for heard, command in cases:
                with self.subTest(heard=heard):
                    step = dict(command, target="none", gesture="none")
                    data = {"emotion": "neutral", "intensity": 0.5,
                            "steps": [step], "lines": {"ko": "해 볼게."}}
                    brain.client.messages.create.return_value = SimpleNamespace(
                        content=[SimpleNamespace(type="text", text=json.dumps(data))],
                        usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn",
                    )
                    reply = poll_output(brain.reply(heard))
                    self.assertEqual(reply["steps"], [step])
                    request = brain.client.messages.create.call_args.kwargs
                    self.assertEqual(request["messages"][-1], {"role": "user", "content": heard})
                    schema = request["output_config"]["format"]["schema"]
                    properties = schema["properties"]["steps"]["items"]["properties"]
                    self.assertIn(command["do"], properties["do"]["enum"])
                    self.assertEqual(properties["action"]["type"], "string")
                    self.assertIn(heard if heard != "12m 떨어져" else "12m", request["system"][0]["text"])

    def test_invalid_distance_refuses_the_plan(self):
        for cm in ("59", "1201", "2m", "60.5", "060", "60\n", 200, None):
            with self.subTest(cm=cm):
                data = {"lines": {"ko": "움직일게."},
                        "steps": [{"do": "keep_distance", "action": cm}, {"do": "come"}]}
                reply = ai._clamp_reply(data, global_mode=True)
                self.assertEqual(reply["steps"], [])
                self.assertEqual(reply["lines"], ai.localized_lines(ai.DISTANCE_REFUSAL))
                self.assertEqual(reply["line"], reply["lines"]["ko"])

    def test_idle_dialog_does_not_send_player_commands(self):
        dialog = [{"npc": npc, "line": "응.", "gesture": "nod", "steps": [{"do": "prone_tv"}]}
                  for npc in ("kwonjun", "itoa")]
        result = clamp_dialog(dialog)
        self.assertIsNotNone(result)
        self.assertTrue(all("steps" not in turn and turn["gesture"] == "nod" for turn in result))


if __name__ == "__main__":
    unittest.main()
