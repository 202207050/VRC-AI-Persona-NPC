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
            ("다른 소파에 앉아", {"do": "go_sit", "target": "other"}),
            ("다른 쪽에 앉아", {"do": "go_sit", "target": "other"}),
            ("지금 자리 말고 앉아", {"do": "go_sit", "target": "other"}),
            ("1인 소파에 앉아", {"do": "go_sit", "target": "single"}),
            ("혼자 앉는 소파에 앉아", {"do": "go_sit", "target": "single"}),
            ("긴 소파에 앉아", {"do": "go_sit", "target": "long"}),
            ("눕는 소파에 앉아", {"do": "go_sit", "target": "long"}),
            ("앉아 있어", {"do": "go_sit"}),
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
                    step = {"target": "none", "gesture": "none", **command}
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
                    self.assertIn(step["target"], properties["target"]["enum"])
                    self.assertEqual(properties["action"]["type"], "string")
                    if command["do"] == "go_sit":
                        phrase = heard.removesuffix("에 앉아").removesuffix(" 앉아")
                        self.assertIn(phrase, request["system"][0]["text"])
                    else:
                        self.assertIn(heard if heard != "12m 떨어져" else "12m", request["system"][0]["text"])

    def test_hug_release_with_live_context(self):
        heard = "이제 놔 줘"
        touch = "[지금: A가 너를 안고 있음]"
        step = {"do": "hug_release", "target": "none", "gesture": "none"}
        data = {"lines": {"ko": "응, 놓을게."}, "steps": [{"do": "hug_release"}]}
        self.assertEqual(ai._clamp_reply(data)["steps"], [step])
        commands = Path(__file__).resolve().parents[3] / "Assets/W04/Scripts/W04Commands.cs"
        self.assertIn('if (doName == "hug_release") return CHugRelease;', commands.read_text(encoding="utf-8"))
        self.assertIn("hug_release", ai.ACTIONS)
        with tempfile.TemporaryDirectory() as folder:
            brain = ai.Brain("fake", Store(folder, migrate_persona=False))
            brain.client = Mock()
            brain.client.messages.create.return_value = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=json.dumps(data))],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn",
            )
            self.assertEqual(poll_output(brain.reply(heard, touch_now=touch))["steps"], [step])
            said = brain.client.messages.create.call_args.kwargs["messages"][-1]["content"]
            self.assertEqual(said, heard + "\n\n" + touch + " (월드가 알려 준 실제 상황이다. 부정하지 않는다.)")
            for options in ({"query": "검색"}, {"recall": True}):
                turn = brain._request_kwargs(heard, "A", touch_now=touch, **options)
                self.assertTrue(turn["messages"][-1]["content"].startswith(said + "\n\n"))
            self.assertEqual(brain.store.history[-2]["content"], heard)
            self.assertNotIn(touch, str(brain.store.history))

    def test_seat_targets_are_specific_to_go_sit(self):
        for invalid in (None, "sofa", "unknown", ["long"]):
            with self.subTest(invalid=invalid):
                self.assertEqual(ai.clamp_steps([{"do": "go_sit", "target": invalid, "gesture": "wave"}]),
                                 [{"do": "go_sit", "target": "none", "gesture": "none"}])
        self.assertEqual(ai.clamp_steps([{"do": "move_to", "target": "long"}, {"do": "sit", "target": "single"}]),
                         [{"do": "move_to", "target": "speaker", "gesture": "none"},
                          {"do": "sit", "target": "none", "gesture": "none"}])

    def test_sit_style_only_for_go_sit(self):
        # TODO 134: lean / upright pass; another value is left out and the NPC still sits (its default way).
        for style in ai.SIT_STYLES:
            self.assertEqual(ai.clamp_steps([{"do": "go_sit", "target": "long", "style": style}]),
                             [{"do": "go_sit", "target": "long", "gesture": "none", "style": style}])
        for invalid in ("swing", "LEAN", "1", None, ["lean"], 1):
            with self.subTest(invalid=invalid):
                self.assertEqual(ai.clamp_steps([{"do": "go_sit", "style": invalid}]),
                                 [{"do": "go_sit", "target": "none", "gesture": "none"}])
        self.assertEqual(ai.clamp_steps([{"do": "sit", "style": "lean"}]),
                         [{"do": "sit", "target": "none", "gesture": "none"}])
        self.assertEqual(ai.clamp_steps([{"do": "go_sit", "target": "long", "action": "lean"}]),
                         [{"do": "go_sit", "target": "long", "gesture": "none", "style": "lean"}])
        self.assertNotIn("style", ai.reply_schema(False, False)["properties"]["steps"]["items"]["properties"])
        self.assertIn('앉는 모양을 말하면 action을 넣는다: "기대 앉아/편하게 앉아/등 기대"는 "lean"', ai.RULES)

    def test_distance_out_of_range_is_clamped_and_said(self):
        # t128 A3: 0.6~12 m ends instead of refusing the whole plan; come still runs.
        come = {"do": "come", "target": "none", "gesture": "none"}
        for cm, kept, note in (("59", "60", "distance_min"), ("0", "60", "distance_min"),
                               ("1201", "1200", "distance_max"), ("060", "60", None)):
            with self.subTest(cm=cm):
                data = {"lines": {"ko": "움직일게.", "en": "Moving.", "ja": "動くね。"},
                        "steps": [{"do": "keep_distance", "action": cm}, {"do": "come"}]}
                reply = ai._clamp_reply(data, global_mode=True)
                self.assertEqual(reply["steps"], [{"do": "keep_distance", "target": "none", "gesture": "none",
                                                   "action": kept}, come])
                self.assertEqual(reply["line"], "움직일게." + (" " + ai.STEP_NOTES[note][0] if note else ""))
                self.assertEqual(reply["lines"]["en"], "Moving." + (" " + ai.STEP_NOTES[note][1] if note else ""))
                self.assertEqual(reply["line"], reply["lines"]["ko"])
        for cm in ("2m", "60.5", "60\n", 200, None):
            with self.subTest(unreadable=cm):
                data = {"lines": {"ko": "움직일게."}, "steps": [{"do": "keep_distance", "action": cm}, {"do": "come"}]}
                reply = ai._clamp_reply(data)
                self.assertEqual((reply["steps"], reply["line"]), ([come], "움직일게."))

    def test_idle_dialog_does_not_send_player_commands(self):
        dialog = [{"npc": npc, "line": "응.", "gesture": "nod", "steps": [{"do": "prone_tv"}]}
                  for npc in ("kwonjun", "itoa")]
        result = clamp_dialog(dialog)
        self.assertIsNotNone(result)
        self.assertTrue(all("steps" not in turn and turn["gesture"] == "nod" for turn in result))


if __name__ == "__main__":
    unittest.main()
