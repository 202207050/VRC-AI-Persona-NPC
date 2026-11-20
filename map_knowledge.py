"""Contextual slice of the map baked by W04MapKnowledge.Bake."""
import json
import math
import re
from pathlib import Path

PATH = Path(__file__).with_name("map_knowledge.json")
MAX_ENTRIES = 30
MAX_RESULTS = 5
TARGETS = ("sofa", "tv", "window", "bridge_view")
NPCS = ("kwonjun", "itoa")
GENERAL = r"어디\s*(?:로\s*)?(?:가\s*볼|가면|갈\s*까)|둘러\s*보|(?:여기|거실|방)에?\s*(?:뭐|무엇)|방\s*구조|길\s*안내|where (?:can|should) (?:i|we) go|look around|room layout|部屋の構造|見て回"
MOVEMENT = r"(?:^|[^가-힣])(?:가\s*(?:봐|줘|세요|자|렴)|가|걸어\s*가(?:\s*봐|\s*줘)?|이동해(?:\s*줘)?|다가가)(?:$|[\s,.!?。！？])|\b(?:go|walk|move)\s+(?:over\s+)?to\b|(?:行って|移動して)"
LOOKUP_RULE = ("[맵 조회] 월드 안 장소나 물건을 설명하거나 찾아갈 때 필요한 위치가 안내에 없으면 "
               "map_lookup에 그 장소나 물건 이름만 적어 조회를 요청할 수 있다. 일반 대화에는 쓰지 않는다.")
GUIDE_RULE = ("## 맵 안내\n이 목록을 근거로 위치와 용도를 짧게 안내한다. 모르는 위치를 지어내지 않는다. "
              "직접 걸어가라는 요청에만 해당 항목에 표시된 move_to 대상을 사용한다. "
              "이동 대상이 표시되지 않은 물건에는 걸어갈 수 있다고 약속하지 않는다. 물건을 집어 오지는 못한다.")


def _short(value):
    return " ".join(value.split())[:100] if isinstance(value, str) else ""


def _vector(value):
    if not isinstance(value, dict):
        return None
    coords = {axis: value.get(axis) for axis in ("x", "y", "z")}
    if all(type(v) in (int, float) and abs(v) <= 100000 and math.isfinite(v) for v in coords.values()):
        return coords
    return None


def load(path=None):
    """Missing or invalid bakes leave normal conversation and existing movement alone."""
    try:
        data = json.loads(Path(path or PATH).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), list):
        return []
    result = []
    for item in data["entries"][:MAX_ENTRIES]:
        if not isinstance(item, dict):
            continue
        name, zone, purpose = (_short(item.get(key)) for key in ("name", "zone", "purpose"))
        aliases = item.get("aliases")
        if not name or not zone or not purpose or _vector(item.get("position")) is None:
            continue
        terms = {name, *(_short(v) for v in aliases[:16])} if isinstance(aliases, list) else {name}
        # Latin aliases respect word boundaries, including TV followed by a Korean particle.
        pattern = "|".join((r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
                            if term.isascii() else re.escape(term))
                           for term in sorted(terms - {""}, key=lambda term: (-len(term), term)))
        # Python's re.escape also escapes blanks and hyphens. Unescape those for JS's /u mode.
        pattern = pattern.replace(r"\ ", " ").replace(r"\-", "-").replace(r"\#", "#").replace(r"\&", "&").replace(r"\~", "~")
        target = item.get("target")
        stands = item.get("stand_positions")
        destinations = {}
        if isinstance(stands, list) and target in TARGETS:
            for stand in stands:
                if isinstance(stand, dict) and stand.get("npc") in NPCS:
                    position = _vector(stand.get("position"))
                    if position is not None:
                        destinations[stand["npc"]] = position
        result.append({"name": name, "zone": zone, "purpose": purpose, "pattern": pattern,
                       "can_press": item.get("can_press") is True,
                       "target": target if destinations else "", "destinations": destinations})
    return result


def select(heard, entries):
    if not heard:
        return []
    ranked = []
    for index, row in enumerate(entries):
        score = max((len(m[0]) for m in re.finditer(row["pattern"], heard, re.I)), default=0)
        if score:
            ranked.append((-score, index, row))
    if ranked:
        return [row for _, _, row in sorted(ranked)[:MAX_RESULTS]]
    return entries[:MAX_RESULTS] if re.search(GENERAL, heard, re.I) else []


def guide(rows, npc):
    if not rows:
        return ""
    lines = [GUIDE_RULE]
    for row in rows:
        press = " · 누를 수 있음" if row["can_press"] else ""
        move = f" · move_to {row['target']}" if npc in row["destinations"] else ""
        lines.append(f"- {row['name']} · {row['zone']} · {row['purpose']}{press}{move}")
    return "\n".join(lines)


def add_move_positions(steps, heard, rows, npc):
    """The old target still selects the stand point in the current Unity runtime."""
    if not rows or not re.search(MOVEMENT, heard, re.I):
        return
    for step in steps:
        if step["do"] != "move_to":
            continue
        for row in rows:
            if step["target"] == row["target"] and npc in row["destinations"]:
                step["position"] = row["destinations"][npc]
                break


def export():
    return {"entries": load(), "limit": MAX_RESULTS, "general": GENERAL, "movement": MOVEMENT,
            "guide": GUIDE_RULE}
