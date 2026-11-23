"""Step vocabulary and field validation shared with the typed-chat relay."""
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "contract" / "actions.json"
PACKAGED = HERE / "actions.generated.json"  # standalone voice checkout, written by export_cloud_prompt
TABLE = json.loads((SOURCE if SOURCE.is_file() else PACKAGED).read_text(encoding="utf-8"))
ROWS = TABLE["actions"]
ENUMS = TABLE["enums"]
ARGS = TABLE["args"]
VOICE = [name for name, row in ROWS.items() if row["voice"]]
CLOUD = [name for name, row in ROWS.items() if row["cloud"]]
TV = [name for name, row in ROWS.items() if row["group"] == "tv"]
DECORATIVE = [name for name, row in ROWS.items() if row.get("decorative")]
SILENT = [name for name, row in ROWS.items() if row.get("silent")]
DISTANCE = ARGS["keep_distance"]["fields"]["action"]
MAX_STEPS, MAX_TV = TABLE["limits"]["steps"], TABLE["limits"]["tv"]
LABELS = {**TABLE["labels"], **{name: row["label"] for name, row in ROWS.items()}}
NONE_MARKERS = frozenset(TABLE["noneMarkers"])


def _strip(value, extra=""):
    def edge(char):
        return char.isspace() or char in extra or unicodedata.category(char) == "Cf"
    start, end = 0, len(value)
    while start < end and edge(value[start]):
        start += 1
    while end > start and edge(value[end - 1]):
        end -= 1
    return value[start:end]


def clean_text(value):
    """value without whitespace or invisible format characters (BOM, zero-width space) at either end.
    reply.ts cleanText is the same."""
    return _strip(value)


def is_none_marker(value):
    """A lone "nothing" mark in a text field ("", "-", "없음", "none"...; table noneMarkers): only a whole
    match after the ends' spaces, invisible characters and periods, case ignored. reply.ts isNoneMarker."""
    return isinstance(value, str) and _strip(value, ".。").casefold() in NONE_MARKERS


def _value(rule, value, npc, replay_keys, notes):
    """None means invalid. Missing and invalid defaults are applied separately by the caller."""
    if rule.get("text"):
        return (clean_text(value) or None) if isinstance(value, str) else None
    if "enum" in rule:
        group = rule["enum"]
        values = TABLE["outfitByNpc"][npc] if group == "outfitByNpc" else ENUMS[group]
        if isinstance(value, str):
            value = rule.get("aliases", {}).get(value, value)
        return value if value in values else None
    if "replayKey" in rule:
        if isinstance(value, str) and re.fullmatch(rule["replayKey"], value):
            return value if replay_keys is None or value in replay_keys else None
        return None
    if "digits" in rule:
        digit = "[0-9]" if rule["digits"] == "ascii" else r"\d"
        if not isinstance(value, str) or not re.fullmatch(digit + "{1,%d}" % rule["length"], value):
            return None
        number = int(value)
    else:
        if type(value) not in (float, int) or not -float("inf") < value < float("inf"):
            return None
        number = value
    if (rule.get("positive") and number <= 0) or (rule.get("nonzero") and number == 0):
        return None
    bounded = min(rule.get("max", float("inf")), max(rule.get("min", -float("inf")), number))
    if bounded != number:
        if not rule.get("clamp"):
            return None
        if "notes" in rule:
            low_note, high_note = rule["notes"]
            notes.append(low_note if number < bounded else high_note)
    elif "boundOnTie" in rule and number == rule[rule["boundOnTie"]]:
        bounded = rule[rule["boundOnTie"]]
    else:
        bounded = number
    if rule.get("normalize"):
        return str(bounded)
    return value if "digits" in rule else bounded


def _step(raw, spec, npc, replay_keys, notes):
    step = {"do": raw["do"], **TABLE["stepDefaults"]}
    if spec.get("omitGesture"):
        del step["gesture"]
    fields = spec["fields"]
    mode = spec.get("select")
    if mode == "exactlyOne":
        fields = {key: rule for key, rule in fields.items() if key in raw}
        if len(fields) != 1:
            return None
    for key, rule in fields.items():
        if rule.get("omittable") and key not in raw:
            continue
        value = _value(rule, raw.get(key, rule.get("missing")), npc, replay_keys, notes)
        if value is None:
            value = rule.get("fallback")
        if value is None:
            if mode == "firstValid" or rule.get("optional"):
                continue
            if "invalidNote" in spec:
                notes.append(spec["invalidNote"])
            return None
        step[key] = value
        if mode == "firstValid":
            return step
    return None if mode == "firstValid" else step


def from_model(item, spec):
    """The model's merged fields -> the inner names (table modelArgs). An inner field already there wins,
    so the old shape ({"percent": 15}) passes unchanged. A blank string is an absent field: the model
    fills fields it does not need with "" (who, action). Spaces and invisible characters alone are blank too."""
    item = {key: value for key, value in item.items() if not (isinstance(value, str) and not clean_text(value))}
    for key, inner in spec.get("modelArgs", {}).items():
        if key not in item:
            continue
        value = item.pop(key)
        if isinstance(inner, dict):  # amount + unit: unit names the inner field
            rule, inner = inner, item.pop(inner["by"], None)
            if inner not in ENUMS[rule["enum"]]:
                continue
        item.setdefault(inner, value)
    return item


def sanitize_steps(raw, actions=None, npc="kwonjun", replay_keys=None):
    allowed = VOICE if actions is None else actions
    notes, candidates, tv = [], [], set()
    dropped = False
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or item.get("do") not in allowed:
            continue
        name = item["do"]
        spec = ARGS[ROWS[name]["args"]] if name in ROWS else ARGS["none"]
        step = _step(from_model(item, spec), spec, npc, replay_keys, notes)
        if step is None:
            continue
        if name in TV:
            if name in tv:
                continue
            if len(tv) == MAX_TV:
                dropped = True
                continue
            tv.add(name)
        candidates.append(step)
    priority = sorted(range(len(candidates)), key=lambda i: candidates[i]["do"] in DECORATIVE)
    if dropped or any(candidates[i]["do"] not in SILENT for i in priority[MAX_STEPS:]):
        notes.append("dropped")
    kept = [candidates[i] for i in sorted(priority[:MAX_STEPS])]
    return sorted(kept, key=lambda s: s["do"] in TV), list(dict.fromkeys(notes))


def step_properties(names):
    targets = [*ENUMS["targets"], *ENUMS["seats"]]
    for clothes in TABLE["outfitByNpc"].values():
        targets.extend(clothes)
    enums = {**ENUMS, "actions": names, "allTargets": list(dict.fromkeys(targets))}
    props = {}
    for name, template in TABLE["schema"].items():
        field = template.copy()
        if "enum" in field:
            field["enum"] = enums[field["enum"]]
        if "description" in field:
            field["description"] = field["description"].format(**DISTANCE)
        props[name] = field
    return props


def optional_count(node, top_only=False):
    """Optional (not required) properties in a JSON schema, every nested object counted (the API's
    structured-output limit counts the whole schema; a step schema used twice counts twice)."""
    if isinstance(node, list):
        return sum(optional_count(value) for value in node)
    if not isinstance(node, dict):
        return 0
    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    count = len([key for key in props if key not in node.get("required", [])])
    return count if top_only else count + sum(optional_count(value) for value in node.values())


def prompt_rules(channel):
    # Keep the author's paragraph order. Several actions are described together in one paragraph.
    paragraphs = []
    for name, row in ROWS.items():
        if not row[channel] or "promptWith" in row:
            continue
        text = row.get("cloudPrompt") if channel == "cloud" else None
        text = text or row.get("prompt") or f'  - {name}: {row["when"]}'
        paragraphs.append((row.get("promptOrder", len(ROWS)), text))
    distance = dict(DISTANCE, min_m=DISTANCE["min"] / 100, max_m=DISTANCE["max"] / 100)
    return "\n".join(text.format(**distance) for _, text in sorted(paragraphs, key=lambda p: p[0]))


def description_lists(names):
    groups = ([key for key in names if key not in TV], [key for key in names if key in TV],
              ENUMS["targets"], ENUMS["gestures"], ENUMS["emotions"])
    return tuple("·".join(LABELS[key] for key in group if key != "none") for group in groups)
