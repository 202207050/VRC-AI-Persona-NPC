"""Store, persona migration, cfg and cloud export: standard-library-only offline tests."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cfg_bridge
import export_cloud_prompt
import persona_ai
import persona_store

T = "2026.11.19 13:44:37 Debug      -  "
LOCAL = "local:test"


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="w04_multi_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.persona = self.root / "kwonjun.md"
        shutil.copyfile(ROOT / "personas/kwonjun.md", self.persona)
        self.store = persona_store.Store(str(self.root / "data"), str(self.persona))
        self.itoa = self.store.for_npc("itoa")
        target = self.root / "itoa.md"
        shutil.copyfile(ROOT / "personas/itoa.md", target)
        self.itoa.persona_path = str(target)
        self.bridge = cfg_bridge.CfgBridge(self.store)


class StoreNpcTests(StoreFixture):
    def test_persona_fallback_and_override(self):
        # Simulate an old installation that has no personas/ directory.
        old = self.root / "legacy"
        old.mkdir()
        (old / "persona.md").write_text("- 이름: 기존 NPC", encoding="utf-8")
        with patch.object(persona_store, "HERE", str(old)), patch.dict(os.environ, {}, clear=True):
            st = persona_store.Store(str(self.root / "fallback"))
            self.assertEqual(st.npc_name(), "기존 NPC")
        with patch.dict(os.environ, {"PERSONA_FILE": str(self.persona)}):
            st = persona_store.Store(str(self.root / "override"))
            self.assertEqual(st.persona_path, str(self.persona))
            self.assertTrue(st.for_npc("itoa").persona_path.endswith(os.path.join("personas", "itoa.md")))


    def test_data_migration_backup_and_restart(self):
        legacy = self.root / "legacy_data"
        legacy.mkdir()
        files = {
            "settings.json": {"keep_history": True, "auto_update": False},
            "impressions.json": {"players": {"Alice": {"note": "kind", "locked": True, "talks": 4}}},
            "memory.json": {"notes": [{"text": "old memo", "by": "user", "at": "2026-11-19"}]},
            "history.json": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        }
        for name, value in files.items():
            (legacy / name).write_text(json.dumps(value), encoding="utf-8")
        st = persona_store.Store(str(legacy), str(self.persona))
        for name in files:
            self.assertEqual((legacy / name).read_bytes(), (legacy / (name + ".bak")).read_bytes())
            self.assertEqual((legacy / name).read_bytes(), Path(st.data_dir, name).read_bytes())
        self.assertEqual(st.players["Alice"]["talks"], 4)
        self.assertEqual(len(st.history), 2)
        other = st.for_npc("itoa")
        self.assertEqual((other.players, other.notes, other.history), ({}, [], []))
        st.add_note("new memo")
        again = persona_store.Store(str(legacy), str(self.persona))
        self.assertEqual([n["text"] for n in again.notes], ["old memo", "new memo"])
        self.assertEqual(json.loads((legacy / "memory.json.bak").read_text())["notes"][0]["text"], "old memo")


    def test_cfg_all_edits_are_per_npc(self):
        send = lambda **req: self.bridge.apply(dict(req, npc="itoa"))
        for req in ({"op": "persona", "text": "- 이름: ITOAR\n- 귤이 좋아"},
                    {"op": "name", "text": "토아"},
                    {"op": "impression", "player": "Alice", "text": "fun"},
                    {"op": "lock", "player": "Alice", "on": True},
                    {"op": "note_add", "text": "one"},
                    {"op": "note_edit", "index": 0, "text": "two", "expect": "one"},
                    {"op": "notes_set", "text": "three", "expect": "two"},
                    {"op": "note_del", "index": 0, "expect": "three"},
                    {"op": "auto", "on": False}, {"op": "keep_history", "on": True}):
            self.assertEqual(send(**req), "")
        self.assertEqual(self.store.npc_name(), "kwonjun")
        self.assertEqual((self.store.players, self.store.notes, self.store.history), ({}, [], []))
        self.assertEqual(self.store.settings, persona_store.DEFAULT_SETTINGS)
        self.assertEqual(self.itoa.npc_name(), "토아")
        self.assertTrue(self.itoa.players["Alice"]["locked"])
        self.itoa.after_reply("Alice", "hi", '{"lines":{"ko":"hey"}}')
        self.store.after_reply("Alice", "kwon hi", '{"lines":{"ko":"kwon hey"}}')
        self.assertEqual(send(op="history_clear"), "")
        self.assertEqual(self.itoa.history, [])
        self.assertEqual(len(self.store.history), 2)
        self.bridge.apply({"op": "note_add", "text": "default"})
        self.assertEqual(self.store.notes[0]["text"], "default")
        self.assertEqual(self.itoa.notes, [])


    def test_cfg_load_and_persona_cut_are_per_npc(self):
        self.itoa.save_persona("- 이름: ITOAR\n" + "귤" * 5900)
        body = json.loads(self.bridge.payload("itoa"))
        self.assertTrue(body["persona_cut"])
        self.assertTrue(self.bridge.apply({"npc": "itoa", "op": "persona", "text": "bad"}))
        self.assertEqual(self.bridge.apply({"op": "persona", "text": "- 이름: kwonjun"}), "")
        self.assertEqual(self.bridge.apply({"npc": "itoa", "op": "load"}), "")
        self.assertEqual(json.loads(self.bridge.payload())["npc"], "ITOAR")
        self.bridge.handle_line("log", LOCAL, T + "[W04AI] CFG_OPEN\n")
        self.assertEqual(json.loads(self.bridge.payload())["npc"], "kwonjun")
        self.bridge.handle_line("log", LOCAL, T + "[W04AI] CFG_OPEN npc=itoa\n")
        self.assertEqual(json.loads(self.bridge.payload())["npc"], "ITOAR")


    def test_dictation_captures_selected_npc(self):
        for target in ("note_add", "impression", "persona_append"):
            self.assertEqual(self.bridge.apply({"npc": "itoa", "op": "dictate", "target": target,
                                               "player": "Alice"}), "")
            self.assertEqual(self.bridge.dictate_target()["npc"], "itoa")
            self.assertIsNone(json.loads(self.bridge.payload("kwonjun"))["dictate"])
            self.assertEqual(self.bridge.dictate_apply("귤 친구"), "")
        self.assertEqual(self.store.notes, [])
        self.assertEqual(self.store.players, {})
        self.assertNotIn("귤 친구", self.store.persona_text())
        self.assertEqual(self.itoa.notes[0]["text"], "귤 친구")
        self.assertEqual(self.itoa.players["Alice"]["note"], "귤 친구")
        self.assertIn("귤 친구", self.itoa.persona_text())


    def test_cloud_export_has_both_and_no_private_memory(self):
        with patch.object(persona_ai, "load_env", side_effect=AssertionError("must not open .env")):
            data = export_cloud_prompt.build_all()
        self.assertEqual(set(data["personas"]), {"kwonjun", "itoa"})
        self.assertEqual(data["personas"]["kwonjun"]["system_fixed"], data["system_fixed"])
        self.assertEqual(data["personas"]["itoa"]["npc_name"], "ITOAR")
        self.assertEqual(data, json.loads((ROOT.parent / "cloud/src/prompt.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
