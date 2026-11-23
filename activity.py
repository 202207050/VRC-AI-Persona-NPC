"""NPC activity record ("활동 기록", act v1): what one turn went through, for the world's settings panel.

The shape is the contract in W04's Tools/contract/activity_v1.json (the cloud relay reads that file
directly; tests/test_golden_contract.py checks that the numbers and kinds here match it and runs its
shared examples). Rows are derived from records the app already keeps (the job's TIMING marks, the AI
reply's info: calls, search_query, links, recall, step notes) at the moment those happen, so nothing is
collected twice.

    act = {"v": 1, "npc": "kwonjun", "turn": 12,
           "rows": [{"d": 0, "k": "heard", "x": "서울 날씨 알려 줘", "ms": 840}, ...]}

A failure / notice row also has "c": an error code of Tools/contract/errors.json (TODO 140), e.g.
{"d": 20100, "k": "error", "x": "AI 서버 응답 시간 초과", "c": "A103"}.

d = ms since the turn started (end of speech), so no clocks have to agree. /cfg rows carry the app PC's
"t": "14:03:21" instead. Everything stays in memory only: nothing here is written to a file.
"""
import collections
import json
import math
import re
import threading
import time

VERSION = 1
NPCS = ("kwonjun", "itoa")
# k -> short Korean label (the world shows its own 3-language words; this is the reference).
KINDS = {
    "listen": "듣는 중", "heard": "받아쓴 말", "think": "생각 중", "search": "검색 중", "source": "출처",
    "recall": "기억 떠올림", "memory": "기억에 적음", "map": "맵 조회", "tv": "화면 정보 확인",
    "reply": "대답", "action": "동작", "drop": "못 한 것", "skip": "넘김", "error": "문제",
    "thought": "속마음", "impression": "인상 고침",
}
MAX_ROWS = 16             # rows in one act
MAX_ACT_BYTES = 1500      # compact UTF-8 JSON of one act
MAX_BODY_BYTES = 7000     # a whole reply body with its act; the world drops bodies over 8192
MAX_TEXT_CHARS = 60       # x, in code points
MAX_D_MS = 120000         # d and ms
HISTORY_ROWS = 20         # rows per NPC kept for /cfg
RECENT_ERRORS = 10        # codes kept for /cfg "errors" (TODO 140)
_CLOCK = re.compile(r"\d\d:\d\d:\d\d")
_CODE = re.compile(r"^[AVWNU][0-9]{3}$")

# Error / notice codes this app produces (TODO 140). The table is W04's Tools/contract/errors.json; this is
# the app's own copy of its rows (level, Korean short text), because the voice app is also a standalone
# repository. tests/test_golden_contract.py checks it against the table, and that every code here is
# really used somewhere in the app. The player never hears a code: it goes to act rows, the log and /cfg.
# --- CODES begin
CODES = {
    "A101": ("info", "응답이 늦어 기다리는 중"),
    "A102": ("warn", "응답 지연, 다시 시도 중"),
    "A103": ("error", "AI 서버 응답 시간 초과"),
    "A104": ("error", "AI 서버 연결 끊김"),
    "A105": ("error", "AI 서버가 바쁨"),
    "A106": ("error", "AI 답 형식 오류"),
    "A107": ("error", "AI 응답 오류"),
    "A108": ("info", "이 부탁은 들어줄 수 없음"),
    "A109": ("info", "옷차림 부탁은 규칙으로 거절"),
    "A110": ("error", "검색 시간 초과"),
    "A111": ("error", "검색 결과를 못 받음"),
    "A121": ("info", "검색이 꺼져 있어 못 찾음"),
    "A122": ("warn", "대화 뒤 기억 정리 실패"),
    "V101": ("info", "점프를 너무 짧게 누름"),
    "V102": ("warn", "마이크 소리가 너무 작음"),
    "V103": ("info", "말소리가 아님"),
    "V104": ("info", "말이 또렷하지 않아 버림"),
    "V105": ("info", "잘못 받아쓴 것 같아 버림"),
    "V106": ("info", "너무 짧거나 알아듣기 어려운 말"),
    "V107": ("warn", "다른 말을 처리 중이라 버림"),
    "V108": ("info", "받아쓰기가 취소돼 버림"),
    "V109": ("error", "음성앱이 이번 말을 처리하지 못함"),
    "V110": ("info", "다른 사람에게 한 말로 보고 넘김"),
    "V111": ("info", "다른 NPC를 불러서 넘김"),
    "V112": ("warn", "허용하지 않은 월드라 무시"),
    "V113": ("info", "마이크 음소거라 녹음 안 함"),
    "V114": ("error", "녹음을 꺼내지 못함"),
    "V115": ("info", "녹음 취소"),
    "W201": ("warn", "가장 가까운 거리로 맞춤"),
    "W202": ("warn", "가장 먼 거리로 맞춤"),
    "W203": ("warn", "옷 메뉴에 없는 것은 뺌"),
    "W204": ("warn", "밝기를 한 단계만 바꿈"),
    "W205": ("info", "백라이트 값을 알려 줌"),
    "W206": ("warn", "나이트 모드 값을 몰라 못 바꿈"),
    "W207": ("warn", "무엇을 바꿀지 몰라 그대로 둠"),
    "W208": ("info", "화면 크기가 아닌 거리로 봄"),
    "W209": ("warn", "한 번에 다 못 해 일부 뺌"),
}
# --- CODES end
# persona_ai.STEP_NOTES key -> code (the table's "note" field)
NOTE_CODES = {"distance_min": "W201", "distance_max": "W202", "outfit": "W203", "backlight_step": "W204",
              "backlight_set": "W205", "night_relative": "W206", "no_target": "W207",
              "distance_guess": "W208", "dropped": "W209"}
# STT reasons (persona_npc_app.judge / checked_transcript; the first word of the reason) -> code
STT_CODES = {"too_short": "V101", "too_quiet": "V102", "no_speech": "V103", "empty": "V103",
             "weak_speech": "V104", "hallucination": "V105", "unclear_short": "V106"}


def code_text(code) -> str:
    return CODES[code][1] if code in CODES else ""


def stt_code(reason) -> str:
    """The code of a failed transcription reason ("too_quiet rms=0.0012" -> V102), else ""."""
    return STT_CODES.get(str(reason or "").split(" ")[0].split("'")[0], "")


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def size(value, dumps=compact) -> int:
    return len(dumps(value).encode("utf-8"))


def clean_text(value) -> str:
    """No tags, no line breaks, at most 60 code points ("…" marks a cut)."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.replace("<", "").replace(">", "").split())
    return text if len(text) <= MAX_TEXT_CHARS else text[:MAX_TEXT_CHARS - 1] + "…"


def _ms(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return int(min(MAX_D_MS, max(0, value)))


def clean_row(row):
    """A row as the contract allows it, or None for a kind outside the list."""
    if not isinstance(row, dict) or row.get("k") not in KINDS:
        return None
    clock = row["t"] if isinstance(row.get("t"), str) and _CLOCK.fullmatch(row["t"]) else None
    out = {} if clock and "d" not in row else {"d": _ms(row.get("d"))}     # a /cfg row has t instead of d
    out.update(k=row["k"], x=clean_text(row.get("x")))
    ms = row.get("ms")
    if not isinstance(ms, bool) and isinstance(ms, (int, float)):
        out["ms"] = _ms(ms)
    code = row.get("c")
    if isinstance(code, str) and _CODE.fullmatch(code):
        out["c"] = code
    if clock:
        out["t"] = clock
    return out


def pack(npc, turn, rows) -> dict:
    """One act: the newest 16 clean rows, then the oldest dropped until it fits 1500 bytes."""
    kept = [row for row in map(clean_row, rows or ()) if row is not None][-MAX_ROWS:]
    act = {"v": VERSION, "npc": npc if npc in NPCS else NPCS[0],
           "turn": turn if isinstance(turn, int) and not isinstance(turn, bool) and turn > 0 else 0,
           "rows": kept}
    while kept and size(act) > MAX_ACT_BYTES:
        kept.pop(0)
    return act


def check(act) -> bool:
    """True when an act is exactly what pack() could have made (the world's reading rule)."""
    return (isinstance(act, dict) and act.get("v") == VERSION and act.get("npc") in NPCS
            and isinstance(act.get("turn"), int) and not isinstance(act.get("turn"), bool) and act["turn"] >= 0
            and isinstance(act.get("rows"), list) and len(act["rows"]) <= MAX_ROWS
            and all(clean_row(row) == row for row in act["rows"]) and size(act) <= MAX_ACT_BYTES)


def fit_body(body: dict, act, limit: int = MAX_BODY_BYTES, dumps=compact) -> dict:
    """body + act, with the oldest act rows left out while the body is over limit bytes, then act itself.
    The reply part is never touched: a record must never make the world throw a reply away."""
    if act is None:
        return body
    rows = list(act.get("rows", ()))
    out = dict(body, act=dict(act, rows=rows))
    while rows and size(out, dumps) > limit:
        rows.pop(0)
    if not rows or size(out, dumps) > limit:
        out.pop("act")                  # an act with no rows is never sent
    return out


class Journal:
    """The last 20 rows per NPC, for the panel's "불러오기" (/cfg). Memory only.
    later(): a row of work that ends AFTER its reply went out (the memory call, t20). It is in /cfg at once
    and rides at the front of that NPC's next turn's act (the next /poll answer)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {npc: collections.deque(maxlen=HISTORY_ROWS) for npc in NPCS}
        self.waiting = {npc: collections.deque(maxlen=MAX_ROWS) for npc in NPCS}

    def add(self, npc, row):
        if npc in self.rows:
            with self.lock:
                self.rows[npc].append(row)

    def later(self, npc, kind, text="", code=None):
        row = clean_row({"t": time.strftime("%H:%M:%S"), "k": kind, "x": text, **({"c": code} if code else {})})
        if npc in self.rows and row is not None:
            if code:
                recent_errors.note(code)
            with self.lock:
                self.rows[npc].append(row)
                self.waiting[npc].append(row)

    def take_later(self, npc) -> list:
        """The later() rows not yet sent, as turn rows (d = 0), oldest first."""
        with self.lock:
            rows = list(self.waiting.get(npc, ()))
            if rows:
                self.waiting[npc].clear()
        return [{"d": 0, **{key: value for key, value in row.items() if key != "t"}} for row in rows]

    def recent(self, npc) -> dict:
        with self.lock:
            rows = list(self.rows.get(npc, ()))
        return {"v": VERSION, "npc": npc if npc in NPCS else NPCS[0], "rows": rows}

    def clear(self):
        with self.lock:
            for rows in (*self.rows.values(), *self.waiting.values()):
                rows.clear()


journal = Journal()


class RecentErrors:
    """/cfg "errors": the newest 10 warn/error codes since the app started, each with the app PC's time of
    its last time and how many times it came. info codes (normal behaviour) are never kept. Memory only."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = collections.OrderedDict()      # code -> {"t", "n"}, newest last

    def note(self, code, at=None):
        if CODES.get(code, ("info",))[0] == "info":
            return
        stamp = time.strftime("%H:%M:%S", time.localtime(time.time() if at is None else at))
        with self.lock:
            item = self.items.pop(code, {"n": 0})
            self.items[code] = {"t": stamp, "n": item["n"] + 1}
            while len(self.items) > RECENT_ERRORS:
                self.items.popitem(last=False)

    def snapshot(self) -> list:
        """[{"c": "A103", "t": "14:03:21", "n": 3}, ...], newest first."""
        with self.lock:
            return [{"c": code, "t": item["t"], "n": item["n"]} for code, item in reversed(self.items.items())]

    def clear(self):
        with self.lock:
            self.items.clear()


recent_errors = RecentErrors()


def note(code):
    """Count a code that has no row of its own (a log-only event) for /cfg; returns the code."""
    recent_errors.note(code)
    return code


def log_code(code) -> str:
    """The voice_log.tsv field for a code."""
    return f"code={code}" if code else ""


class Turn:
    """The rows of one turn. Each row also goes to the journal once, when it is added."""

    def __init__(self, npc, seq, start=None, sink=journal):
        self.npc, self.seq = npc, seq
        self.start = time.monotonic() if start is None else start
        self.rows, self.sink, self._seen, self._later_taken = [], sink, {}, False

    def add(self, kind, text="", at=None, ms=None, code=None):
        key = kind, text
        if key in self._seen:           # an interim act already carried it: only a time taken is added
            if ms is not None:
                for row in self._seen[key]:
                    row.setdefault("ms", _ms(ms))
            return
        at = time.monotonic() if at is None else at
        row = clean_row({"d": round((at - self.start) * 1000), "k": kind, "x": text,
                         **({} if ms is None else {"ms": ms}), **({"c": code} if code else {})})
        if row is None:
            return
        if code:
            recent_errors.note(code, time.time() - (time.monotonic() - at))
        self.rows.append(row)
        self._seen[key] = [row]
        if self.sink is not None:
            stamp = time.strftime("%H:%M:%S", time.localtime(time.time() - (time.monotonic() - at)))
            saved = {name: value for name, value in dict(row, t=stamp).items() if name != "d"}
            self._seen[key].append(saved)
            self.sink.add(self.npc, saved)

    def add_code(self, code, kind=None, text=None, at=None, ms=None):
        """A row for an error / notice code: x = the table's short text unless given; kind = "error" for
        warn / error codes and "drop" for info codes unless given."""
        if kind is None:
            kind = "drop" if CODES.get(code, ("info",))[0] == "info" else "error"
        self.add(kind, code_text(code) if text is None else text, at, ms, code=code)

    def act(self) -> dict:
        """The act sent with a reply. The first one also carries what finished after the previous reply
        (Journal.later: the memory call's rows), at the front."""
        if self.sink is not None and not self._later_taken:
            self._later_taken = True
            self.rows[:0] = self.sink.take_later(self.npc)
        return pack(self.npc, self.seq, [dict(row) for row in self.rows])

    def add_steps(self, info, ai_start):
        """search / recall / map rows from what the AI reply already reports (its calls' ms say when)."""
        at = ai_start
        for call in info.get("calls", ()):
            kind = call.get("kind")
            if kind == "map":
                self.add("map", "다시 조회", at)
            elif kind == "recall":
                self.add("recall", "", at, info.get("recall_ms"))
            elif kind == "search" and info.get("search_query"):
                self.add("search", info["search_query"], at)
            at += (call.get("ms") or 0) / 1000
        if info.get("map_used"):
            self.add("map", " · ".join(info["map_used"][:2]), ai_start)
        if info.get("search_query"):        # the fake AI has no calls
            self.add("search", info["search_query"], ai_start)
        if info.get("recall"):
            self.add("recall", "", ai_start, info.get("recall_ms"))

    def add_result(self, info, ai_start, ai_end, step_text=None):
        """Everything one finished AI reply reports, in the order the player would see it."""
        self.add_steps(info, ai_start)
        for link in (info.get("links") or ())[:3]:
            self.add("source", link.get("title", ""), ai_end)
        code = info.get("code") or ("A107" if info.get("error") else None)   # a brain without codes
        if code:
            # A107 keeps the exception's class name (never its message): it is the "anything else" bucket.
            name = (info.get("error") or "").split(":")[0]
            self.add_code(code, text=f"{code_text(code)} · {name}" if code == "A107" and name else None, at=ai_end)
        for key in info.get("step_notes") or ():
            text = step_text(key) if step_text else key
            if text:
                self.add("drop", text, ai_end, code=NOTE_CODES.get(key))
