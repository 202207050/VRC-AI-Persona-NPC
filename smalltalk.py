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
DIALOG_REQUEST = re.compile(
    r"(?:둘\s*이(?:서)?|두\s*NPC(?:가|는)?|NPC\s*(?:들)?끼리|너희\s*끼리|서로)\s*"
    r"(?:먼저\s*)?(?:대화|얘기|이야기|잡담)(?:를)?\s*(?:한번\s*|한\s*번\s*)?"
    r"(?:해(?:\s*(?:봐요?|줘|줄래|볼래|주세요))?|하(?:자|라고|도록))(?=$|[\s.!?,…！？。])", re.IGNORECASE
)
REACTION_NAMES = {"kwonjun": ("kwonjun", "권준", "권준이"), "itoa": ("itoa", "itoar", "이토아")}


def requested(text):
    return DIALOG_REQUEST.search(text) is not None


def _vocative(line, names, bare_start=False):
    """True when one of names is used to call someone: '이토아, 들어 봐', '이토아야', '야 이토아'.
    Plain mentions such as '이토아가 그러더라' or '이토아는 뭐 좋아해?' are not. bare_start: a name that
    opens the sentence followed by a space ('이토아 이리 와') also counts."""
    names = "|".join(re.escape(name) for name in sorted(set(filter(None, names)), key=len, reverse=True))
    bare = r"|\s+" if bare_start else ""
    start = rf"^\s*(?:야[ ,]+)?(?:{names})(?:(?:아|야|\s*씨)(?=\s|[,.!?…！？，]|$)|\s*[,，:!?！？]|${bare})"
    inside = rf"[,.!?…，]\s*(?:야[ ,]+)?(?:{names})(?:(?:아|야|\s*씨)(?=\s|[,.!?…！？，]|$)|\s*[,，:!?！？]|$)"
    return re.search(start + "|" + inside, line, re.IGNORECASE) is not None


def addressed(line, other):
    return _vocative(line, (other.npc_name(), *REACTION_NAMES[other.npc]))


def called(text, names):
    """A player's words call this NPC by name (not just mention it). See _vocative."""
    return _vocative(text, names, bare_start=True)


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

    def reply(self, brain, still_idle, global_mode=False, content_level=0, requested_by_player=False, opener=None):
        """Return (world payload, source), with a scripted fallback for a failed model call."""
        dialog = None
        source = "script"
        if still_idle() and getattr(brain, "client", None) is not None:
            try:
                reservation = {"forced": True} if requested_by_player else {}
                if self.store.reserve_smalltalk_ai(self.rng.random(), **reservation) and still_idle():
                    profiles = [self.store.for_npc(npc) for npc in NPC_IDS]
                    options = {"global_mode": True} if global_mode else {}
                    if content_level:
                        options["content_level"] = content_level
                    if opener:
                        options["opener"] = opener
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
        if opener:
            # A normal answer already delivered the opening turn. Reply once to it.
            partner = next(npc for npc in NPC_IDS if npc != opener["npc"])
            if dialog:
                dialog = [next(row for row in dialog if row["npc"] == partner)]
            else:
                lines = ("응, 나 듣고 있어.", "Yeah, I'm listening.", "うん、聞いてるよ。")
                dialog = [{"npc": partner, "line": lines[0], "emotion": "neutral", "intensity": 0.5,
                           "gesture": "nod", "lines": dict(zip(("ko", "en", "ja"), lines if global_mode else lines[:1]))}]
        return {"npc": (dialog or [{}])[0].get("npc", NPC_IDS[0]),
                "dialog": dialog or self.scripted(global_mode), "smalltalk": True}, source

    def remember(self, row, previous=None):
        npc = row["npc"]
        partner = self.store.for_npc(next(key for key in NPC_IDS if key != npc))
        current = self.store.for_npc(npc)
        current.record_npc_exchange(partner.npc_name(), (previous or {}).get("line", ""), row["line"])
        if previous:
            partner.record_npc_exchange(current.npc_name(), row["line"], "")
