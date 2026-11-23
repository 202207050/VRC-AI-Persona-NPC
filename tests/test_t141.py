"""TODO 141, offline: calmer adult speech. The default personas, fixed lines and small talk have no tangerine
habit, no "~" and no "!"; saved personas that still hold an old default line are moved to the new line,
lines the user changed stay. No API call.

    .venv\\Scripts\\python.exe -X utf8 tests\\test_t141.py
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import persona_ai as ai
import persona_store

CHILDISH = re.compile(r"귤|みかん|tangerine|[~〜]|우와|짱", re.IGNORECASE)

# The ITOAR persona as shipped before t141 (only the lines that matter), plus one line the user changed.
OLD_ITOA = """# 이토아 성격 설정

## 누구인가
- 이름: 이토아
- 밤의 도시가 내려다보이는 높은 층 거실에 사는 밝고 호기심 많은 20대 초반 여성. 후드 차림.
- 함께 사는 권준은 장난기가 있고 밤 풍경과 영화를 좋아하는 친구다.

## 말투와 성격
- 친근하고 편한 반말. 말끝을 귀엽게 늘이거나 아이 같은 말투를 쓰지 않는다. 장난기가 조금 있지만 상대를 놀리거나 무시하지는 않는다.
- 호기심이 많아서 손님에게 이것저것 묻는 걸 좋아한다.
- 귤을 아주 좋아한다. 손님과 귤을 나누고 새로운 이야기를 듣는 걸 좋아한다.
- 귤을 아주 좋아한다. 내가 직접 고친 줄.
"""
OLD_KWONJUN = ("# 권준 성격 설정 (예시)\n- 이름: 권준\n"
               "- 함께 사는 이토아는 밝고 호기심이 많으며 귤을 좋아하는 친구다.\r\n"
               "- 밤 풍경, 영화, 음악, 소파에서 뒹굴며 쉬는 걸 좋아한다.\n")


def fixed_lines():
    yield from ai.REFUSAL.values()
    yield from ai.INTERIM_LINES["wait"]["kwonjun"] + ai.INTERIM_LINES["wait"]["itoa"]
    yield ai.INTERIM_LINES["search"]
    yield ai.INTERIM_LINES["recall"]
    yield ai.VOICE_ERROR
    yield ai.VOICE_SLOW
    yield ai.SEARCH_ERROR


class T141(unittest.TestCase):
    def test_default_personas_are_plain(self):
        for name in ("personas/kwonjun.md", "personas/itoa.md", "persona.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("귤", text, name)
            self.assertNotIn("호기심이 많아서", text, name)
            self.assertIn("담백하게", text, name)
            # the shipped files are already today's defaults: the migration changes nothing in them
            store = persona_store.Store.__new__(persona_store.Store)
            store.npc = "itoa" if "itoa" in name else "kwonjun"
            self.assertEqual(store._renamed_persona(text), text, name)

    def test_fixed_lines_are_plain(self):
        for line in fixed_lines():
            self.assertEqual(len(line), 3)
            for text in line:
                self.assertIsNone(CHILDISH.search(text), text)
                self.assertNotIn("!", text)
                self.assertNotIn("！", text)

    def test_small_talk_is_plain(self):
        dialogs = json.loads((ROOT / "smalltalk.json").read_text(encoding="utf-8"))["kwonjun,itoa"]
        self.assertEqual(len(dialogs), 45)
        for dialog in dialogs:
            for item in dialog:
                for text in (item["line"], item["lines"]["en"], item["lines"]["ja"]):
                    self.assertIsNone(CHILDISH.search(text), text)
                self.assertNotIn("!", item["line"])
                self.assertLessEqual(item["intensity"], 0.5)
                self.assertNotEqual(item["emotion"], "surprised")

    def test_cloud_export_has_the_same_refusal(self):
        prompt = json.loads((ROOT.parent / "cloud" / "src" / "prompt.json").read_text(encoding="utf-8"))
        self.assertEqual(prompt["error_lines"]["refusal"],
                         {npc: ai.localized_lines(text) for npc, text in ai.REFUSAL.items()})
        for persona in prompt["personas"].values():
            self.assertNotIn("귤", persona["system_fixed"])

    def test_saved_persona_old_default_lines_move(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "personas").mkdir()
            (root / "personas" / "itoa.md").write_text(OLD_ITOA, encoding="utf-8")
            (root / "personas" / "kwonjun.md").write_bytes(OLD_KWONJUN.encode("utf-8"))
            with patch.object(persona_store, "HERE", str(root)), patch.dict("os.environ", {}, clear=True):
                store = persona_store.Store(str(root / "data"))
                itoa = store.for_npc("itoa")
            saved = (root / "personas" / "itoa.md").read_text(encoding="utf-8")
            self.assertEqual(saved, itoa.persona_text())
            self.assertIn("- 이름: ITOAR", saved)
            self.assertIn("- 밤의 도시가 내려다보이는 높은 층 거실에 사는 20대 초반 여성. 후드 차림.", saved)
            self.assertIn("- 함께 사는 kwonjun은 차분하고 밤 풍경과 영화를 좋아하는 친구다.", saved)
            self.assertIn("담백하게", saved)
            self.assertIn("질문은 대화가 이어질 때 가끔만 한다.", saved)
            self.assertIn("- 조용한 음악을 틀어 두고 밤 풍경을 보거나 산책하는 걸 좋아한다.", saved)
            # the line the user changed is kept word for word
            self.assertIn("- 귤을 아주 좋아한다. 내가 직접 고친 줄.", saved)
            self.assertEqual(saved.count("귤"), 1)
            # a Windows (CRLF) copy is read with universal newlines and written back with "\n"
            kwonjun = (root / "personas" / "kwonjun.md").read_bytes().decode("utf-8")
            self.assertEqual(kwonjun, "# kwonjun 성격 설정 (예시)\n- 이름: kwonjun\n"
                                      "- 함께 사는 ITOAR는 차분하고 다정한 친구다.\n"
                                      "- 밤 풍경, 영화, 음악, 소파에서 쉬는 걸 좋아한다. "
                                      "먹을 것이나 장난감 같은 소재를 습관처럼 꺼내지 않는다.\n")

    def test_lines_only_match_whole_lines(self):
        store = persona_store.Store.__new__(persona_store.Store)
        store.npc = "itoa"
        edited = "- 호기심이 많아서 손님에게 이것저것 묻는 걸 좋아한다. 특히 영화 얘기.\n"
        self.assertEqual(store._renamed_persona(edited), edited)
        self.assertEqual(store._renamed_persona("  " + edited), "  " + edited)


if __name__ == "__main__":
    unittest.main()
