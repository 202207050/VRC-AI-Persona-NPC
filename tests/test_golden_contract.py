"""Run the same offline cases as the cloud relay (test/golden.test.ts): the reply rules
(contract/golden_replies.json) and the activity record format (contract/activity_v1.json)."""
import json
from pathlib import Path
import re
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[2]
VOICE = TOOLS / "voice"
sys.path.insert(0, str(VOICE))
import activity
import persona_ai as ai
import export_cloud_prompt as cloud
from persona_ai import STEP_NOTES, _clamp_reply, resolve_relative

CASES = json.loads((TOOLS / "contract" / "golden_replies.json").read_text(encoding="utf-8"))
ACTIVITY = json.loads((TOOLS / "contract" / "activity_v1.json").read_text(encoding="utf-8"))
ERRORS = json.loads((TOOLS / "contract" / "errors.json").read_text(encoding="utf-8"))
MEMORY = json.loads((TOOLS / "contract" / "memory_v1.json").read_text(encoding="utf-8"))


def run_memory(fn, value):
    import cfg_bridge
    import persona_npc_app as app
    if fn == "digest":
        return cfg_bridge.digest(value)
    m = app.CHAT_REQ_RE.search(value)
    return m and {"npc": m.group(1), "id": m.group(2), "by": m.group(3), "q": m.group(4)}


def check_case(self, case):
    # resolve_relative with the case's TV state (voice-only cases), else none: what the cloud does.
    reply = _clamp_reply(case["data"], case.get("global", False), heard=case["heard"], npc=case["npc"],
                         replay_keys=case.get("replay_keys"), tv_before=case.get("tv_before", False))
    reply = resolve_relative(reply, case.get("state"))
    self.assertEqual({key: reply[key] for key in case["expected"]}, case["expected"])


def run_activity(fn, value):
    if fn == "row":
        return activity.clean_row(value)
    if fn == "pack":
        return activity.pack(value["npc"], value["turn"], value["rows"])
    if fn == "body":
        return activity.fit_body(value["body"], value["act"], value["limit"])
    return activity.check(value)


class GoldenContractTests(unittest.TestCase):
    def test_action_table_matches_protocol(self):
        table = json.loads((TOOLS / "contract" / "actions.json").read_text(encoding="utf-8"))
        self.assertEqual(table, json.loads(ai.contract.PACKAGED.read_text(encoding="utf-8")))
        rows = table["actions"]
        self.assertEqual([k for k, r in rows.items() if r["voice"]], ai.ACTIONS)
        self.assertEqual([k for k, r in rows.items() if r["cloud"]], cloud.cloud_schema()["properties"]["steps"]["items"]["properties"]["do"]["enum"])
        self.assertEqual([k for k, r in rows.items() if r["group"] == "tv"], ai.TV_ACTIONS)
        self.assertEqual([k for k, r in rows.items() if r.get("decorative")], list(ai.DECORATIVE_ACTIONS))
        self.assertEqual({k: ai.SELF_LABELS[k] for k in rows}, {k: r["label"] for k, r in rows.items()})
        self.assertEqual(table["enums"]["styles"], ai.SIT_STYLES)
        self.assertEqual(table["outfitByNpc"], {k: list(v) for k, v in ai.OUTFIT_TARGETS.items()})
        distance = table["args"]["keep_distance"]["fields"]["action"]
        self.assertEqual((distance["min"], distance["max"]), (ai.MIN_DISTANCE_CM, ai.MAX_DISTANCE_CM))
        self.assertEqual((table["limits"]["steps"], table["limits"]["tv"]), (ai.MAX_STEPS, ai.MAX_TV_STEPS))
        self.assertEqual(set(table["args"]), {r["args"] for r in rows.values()})

    def test_schema_optional_fields_stay_few(self):
        # t18 (D121): optional fields multiply the API's schema preparation time (a 27-field schema is
        # rejected, 11 step fields took 36 s). Nothing optional at the top; voice <= 6 (t18c), cloud <= 12 in all.
        # t19: small talk (Global ON and OFF) and memory compaction are separate schemas: none optional.
        # t20: the reply lost impression / memory_note; the memory call is a separate schema: none optional.
        from itertools import product
        count = ai.contract.optional_count
        for flags in product((False, True), repeat=5):
            *args, global_mode = flags
            schema = ai.reply_schema(*args)
            self.assertNotIn("impression", schema["properties"])
            self.assertNotIn("memory_note", schema["properties"])
            schema = ai.global_lines(schema) if global_mode else schema
            with self.subTest(flags=flags):
                self.assertEqual(count(schema, top_only=True), 0)
                self.assertLessEqual(count(schema), 6)
        for name, schema in (("smalltalk", ai.SMALLTALK_SCHEMA), ("smalltalk+global", ai.global_lines(ai.SMALLTALK_SCHEMA)),
                             ("compact", ai.COMPACT_SCHEMA), ("memory", ai.MEMORY_SCHEMA)):
            with self.subTest(schema=name):
                self.assertEqual(count(schema), 0)
        prompt = cloud.build()
        search = dict(prompt["schema"], required=["search", *prompt["schema"]["required"]],
                      properties={"search": prompt["search"]["schema_property"], **prompt["schema"]["properties"]})
        for schema in (prompt["schema"], search):
            self.assertEqual(count(schema, top_only=True), 0)
            self.assertLessEqual(count(schema), 12)

    def test_numeric_boundary_wire_format(self):
        # The world receives JSON text. At an exact bound the previous clamp sometimes kept a float.
        for name, key, value, extra, encoded in [
            ("tv_seek", "seconds", 86400.0, {}, "86400.0"),
            ("tv_seek", "seconds", -86400.0, {}, "-86400"),
            ("tv_size_set", "inches", 1000000.0, {}, "1000000.0"),
            ("tv_size_set", "percent", -0.0, {}, "-0.0"),
            ("tv_backlight", "percent", -0.0, {}, "0"),
            ("emote", "duration", 10.0, {"emotion": "happy"}, "10"),
        ]:
            with self.subTest(name=name, value=value):
                step, = ai.clamp_steps([{"do": name, key: value, **extra}])
                self.assertEqual(json.dumps(step[key]), encoded)

    def test_activity_definition(self):
        # activity.py keeps its own copy (the voice app is also a standalone repository): it must match.
        limits = ACTIVITY["limits"]
        self.assertEqual(activity.VERSION, ACTIVITY["v"])
        self.assertEqual(list(activity.NPCS), ACTIVITY["npcs"])
        self.assertEqual(activity.KINDS, ACTIVITY["kinds"])
        self.assertEqual((activity.MAX_ROWS, activity.MAX_ACT_BYTES, activity.MAX_BODY_BYTES,
                          activity.MAX_TEXT_CHARS, activity.MAX_D_MS, activity.HISTORY_ROWS),
                         (limits["rows"], limits["act_bytes"], limits["body_bytes"], limits["text_chars"],
                          limits["d_ms"], limits["history_rows"]))
        self.assertLess(limits["body_bytes"], limits["world_body_bytes"])
        produced = {kind for group in ACTIVITY["producers"].values() for kind in group}
        self.assertEqual(produced, set(ACTIVITY["kinds"]))
        self.assertEqual(activity._CODE.pattern, ACTIVITY["codes"]["pattern"])

    def test_error_codes_match_the_table(self):
        # TODO 140: the app's copy (activity.CODES / NOTE_CODES) is exactly the table's app rows.
        app = {row["code"]: row for row in ERRORS["codes"] if "app" in row["where"]}
        self.assertEqual({code: (row["level"], row["text"]["ko"]) for code, row in app.items()}, activity.CODES)
        self.assertEqual({row["note"]: code for code, row in app.items() if row.get("note")}, activity.NOTE_CODES)
        self.assertEqual(set(activity.NOTE_CODES), set(STEP_NOTES), "every step note has a code")
        self.assertLessEqual(set(activity.STT_CODES.values()), set(activity.CODES))

    def test_memory_definition(self):
        # t20: memory_v1.json is what the world worker reads; its numbers are the app's
        import cfg_bridge
        import persona_npc_app as app
        import persona_store
        chat, cfg, digest = MEMORY["chat_req"], MEMORY["cfg_memory"]["limits"], MEMORY["digest"]
        self.assertEqual(chat["regex"], app.CHAT_REQ_RE.pattern)
        self.assertEqual((chat["limits"]["q_chars"], chat["limits"]["by_chars"]),
                         (app.CHAT_MAX_CHARS, persona_store.MAX_NAME_CHARS))
        self.assertEqual(MEMORY["npcs"], list(activity.NPCS))
        self.assertEqual((cfg["people"], cfg["notes"], cfg["persona_lines"], cfg["card_chars"], cfg["profile_lines"],
                          cfg["cfg_bytes"]),
                         (cfg_bridge.MEMORY_PEOPLE, cfg_bridge.MEMORY_NOTES, cfg_bridge.MEMORY_PERSONA_LINES,
                          cfg_bridge.CARD_CHARS, persona_store.PROFILE_LINES, cfg_bridge.MAX_CFG_BYTES))
        self.assertEqual(digest["max_bytes"], cfg_bridge.DIGEST_MAX_BYTES)
        self.assertTrue(digest["format"][0].endswith(cfg_bridge.DIGEST_HEAD))
        self.assertLessEqual(set(MEMORY["activity"]["kinds"]) - {"error"}, set(ACTIVITY["kinds"]))
        self.assertIn("A122", activity.CODES)

    def test_every_app_code_is_used_and_none_is_unknown(self):
        # the CODES block is the copy, not a use: every code must also appear somewhere else in the app
        text = {}
        for path in VOICE.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            if path.name == "activity.py":
                start, end = source.index("# --- CODES begin"), source.index("# --- CODES end")
                source = source[:start] + source[end:]
            text[path.name] = source
        used = {code for source in text.values() for code in re.findall(r"[\"']([AVWNU][0-9]{3})[\"']", source)}
        self.assertEqual(used, set(activity.CODES))


for index, case in enumerate(CASES):
    if case.get("only") != "cloud":
        setattr(GoldenContractTests, f"test_golden_{index:03}",
                lambda self, case=case: check_case(self, case))

for case in ACTIVITY["cases"]:
    setattr(GoldenContractTests, f"test_activity_{case['id']}",
            lambda self, case=case: self.assertEqual(run_activity(case["fn"], case["in"]), case["out"]))

for case in MEMORY["cases"]:
    setattr(GoldenContractTests, f"test_memory_{case['id']}",
            lambda self, case=case: self.assertEqual(run_memory(case["fn"], case["in"]), case["out"]))

if __name__ == "__main__":
    unittest.main()
