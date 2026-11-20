"""Local continuation for a world button whose text cannot be put into a runtime VRCUrl."""
import json
import re
import threading
import time

NONCE = re.compile(r"-?[0-9]+:[0-9]+:[0-9]+\Z")


def read_event(line, marker):
    if marker not in line:
        return None
    try:
        data = json.loads(line.split(marker, 1)[1])
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("npc") not in ("kwonjun", "itoa"):
        return None
    context = data.get("context", [])
    if not isinstance(context, list) or len(context) != 2 or not all(
            isinstance(item, str) and len(item) <= 200 for item in context):
        return None
    if marker.endswith("CHOICE_INPUT "):
        ident, text = data.get("id"), data.get("text")
        if not isinstance(ident, str) or len(ident) > 60 or not NONCE.fullmatch(ident):
            return None
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            return None
    return data


class ChoiceMailbox:
    def __init__(self):
        self.ready = threading.Condition()
        self.results = {}

    def put(self, npc, reply):
        with self.ready:
            self.results[npc] = (time.monotonic(), reply)
            self.ready.notify_all()

    def take(self, npc, timeout=18):
        expires = time.monotonic() + timeout
        with self.ready:
            while True:
                got = self.results.pop(npc, None)
                now = time.monotonic()
                if got and now - got[0] <= 8:
                    return got[1]
                if now >= expires:
                    return {}
                self.ready.wait(expires - now)
