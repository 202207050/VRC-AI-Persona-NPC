"""Local NPC-to-NPC exchanges for the app's existing /poll queue."""
from collections import deque
import json
from pathlib import Path
import random
import re

from persona_ai import _clamp_reply
from persona_store import NPC_IDS

SCRIPT = Path(__file__).with_name("smalltalk.json")
REQUEST_RE = re.compile(r"\[W04AI\] NPC_CHAT_REQ[ \t]+npcs=([^\s]+)[ \t]+watcher=([^\r\n]+)\Z")


def parse_request(line: str):
    match = REQUEST_RE.search(line.rstrip("\r\n"))
    if match:
        pair = match[1].split(",")
        watcher = match[2].strip()
        if len(pair) == len(NPC_IDS) and set(pair) == set(NPC_IDS) and watcher:
            return NPC_IDS, watcher
    return None


def clamp_dialog(dialog, global_mode=False):
    """Reject malformed turns and use the same line, emotion and gesture rules as player replies."""
    if not isinstance(dialog, list) or not 2 <= len(dialog) <= 4:
        return None
    result = []
    for i, item in enumerate(dialog):
        npc = NPC_IDS[i % len(NPC_IDS)]
        if not isinstance(item, dict) or item.get("npc") != npc:
            return None
        translations = item.get("lines")
        lines = dict(translations) if isinstance(translations, dict) else {}
        ko = item.get("line", lines.get("ko"))
        if not isinstance(ko, str):
            return None
        lines["ko"] = ko
        reply = _clamp_reply({"lines": lines, "emotion": item.get("emotion"),
                              "intensity": item.get("intensity", 0.5),
                              "steps": [{"do": "gesture", "gesture": item.get("gesture")}]}, global_mode)
        if not reply["line"]:
            return None
        result.append({"npc": npc, "line": reply["line"], "emotion": reply["emotion"],
                       "intensity": reply["intensity"], "gesture": reply["steps"][0]["gesture"]})
        if len(reply["lines"]) > 1:
            result[-1]["lines"] = reply["lines"]
    return result


class SmallTalk:
    def __init__(self, store, rng=None):
        self.store = store
        self.rng = rng if rng is not None else random.Random()
        pair = ",".join(NPC_IDS)
        self.scripts = [dialog for raw in json.loads(SCRIPT.read_text(encoding="utf-8"))[pair]
                        if (dialog := clamp_dialog(raw, global_mode=True)) is not None]
        if not self.scripts:
            raise ValueError("no valid small talk scripts")
        self.recent = deque(maxlen=min(10, len(self.scripts) - 1))

    def scripted(self, global_mode=False):
        index = self.rng.choice([i for i in range(len(self.scripts)) if i not in self.recent])
        self.recent.append(index)
        return clamp_dialog(self.scripts[index], global_mode)

    def reply(self, brain, still_idle, global_mode=False, content_level=0):
        """Return (world payload, source), with a scripted fallback for a failed model call."""
        dialog = None
        source = "script"
        if still_idle() and getattr(brain, "client", None) is not None:
            try:
                if self.store.reserve_smalltalk_ai(self.rng.random()) and still_idle():
                    profiles = [self.store.for_npc(npc) for npc in NPC_IDS]
                    options = {"global_mode": True} if global_mode else {}
                    if content_level:
                        options["content_level"] = content_level
                    dialog = clamp_dialog(brain.smalltalk(profiles, **options), global_mode)
                    if global_mode and dialog and any(
                        not row.get("lines", {}).get(lang) for row in dialog for lang in ("en", "ja")
                    ):
                        dialog = None
                    source = "ai" if dialog else "fallback"
            except Exception:
                source = "fallback"
        if not still_idle():
            return None, source
        return {"npc": NPC_IDS[0], "dialog": dialog or self.scripted(global_mode), "smalltalk": True}, source
