"""VRC AI Persona NPC - companion app on the speaker's PC.

Look at the NPC from close by and just talk (listen mode "gaze", the default) -> this app hears the
microphone and cuts it into utterances -> Whisper turns each into text -> Claude answers as the NPC's
persona (persona_ai.py, key in .env) -> the world, which is waiting on http://127.0.0.1:18770/poll,
gets the reply and the NPC says it.
Without a key, or with --ai off, the NPC just repeats what it heard ("들은 말: ...").

How it knows when to listen: the world writes lines into VRChat's output_log (see README "월드 쪽
규칙") and this app tails that file.
  - gaze mode: "[W04AI] LISTEN_ON" while the NPC is near the middle of the player's view and close,
    "[W04AI] LISTEN_OFF" when not. While it is on, a voice activity detector (an energy gate with an
    adaptive noise floor, UtteranceDetector) starts an utterance when speech begins (with 0.5 s from
    just before), ends it after 1.3 s of silence by default or 15 s, and drops blips under 0.21 s of speech.
    A LISTEN_OFF in the middle of an utterance lets that utterance finish. While one utterance is
    being answered, new speech is usually dropped. A fragment can be continued, and speech during a
    web search is queued for the worker.
  - push-to-talk: "[W04AI] PTT_DOWN" / "[W04AI] PTT_UP" (hold jump). The app always accepts these
    lines, but the sample world only writes them when its W04Probe "jumpTalk" switch is on, and that
    switch is OFF by default (jump is a normal jump). With --listen ptt (or W04_LISTEN_MODE=ptt in
    .env) LISTEN lines are ignored and jump is the only way to talk: use it only with a world that
    has jumpTalk on, or nothing will start an utterance.

Run (double-click run.cmd, or):
    .venv\\Scripts\\python.exe persona_npc_app.py            default microphone
    .venv\\Scripts\\python.exe persona_npc_app.py --list     list microphones
    .venv\\Scripts\\python.exe persona_npc_app.py --device 3 use microphone #3

Settings page: the console prints http://127.0.0.1:18770/settings?t=<random token> at startup (or
double-click settings.cmd). It shows and edits the persona, the NPC's impression of each player and its
memory (persona_store.py). The token changes on every run; without it the page and its data answer 403,
because any VRChat world may send GET requests to 127.0.0.1 when "Allow Untrusted URLs" is on.

In-world settings panel: the world writes "[W04AI] CFG_OPEN" / "[W04AI] CFG_SET {json}" to the log and
reads GET /cfg (cfg_bridge.py). No token there; instead /cfg answers only for 20 s after CFG_OPEN, and
[W04AI] lines count only from a log whose current world is allowed (Build & Test or the combined W04_WORLD_IDS allowlist).

The microphone is kept open while the app runs; only the last 0.5 s stays in memory, and audio is
looked at (and kept, for the length of one utterance) only while LISTEN is on or jump is held.
Nothing is written to disk unless --save-clips is given.
"""
import argparse
import collections
import glob
import hmac
import json
import math
import os
import queue
import re
import secrets
import socket
import statistics
import sys
import threading
import time
import traceback
import wave
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import PurePath
from urllib.parse import parse_qs

import numpy as np

import activity
import cfg_bridge
import local_commands
import osc_mute
from choice_relay import ChoiceMailbox, read_event
from voice_options import DEFAULT_MODEL, MODELS, hint_prompt, parse_hints, replace_hints, whisper_beam_size
from datetime import datetime

import persona_ai
import persona_store
import search_links
import smalltalk
import tv_history

PORT = 18770
MARKER = "[W04AI]"                 # prefix the world puts on its output_log lines
RATE = 16000                       # Whisper works on 16 kHz mono
PRE_ROLL_S = 0.5                   # audio kept from just before the press (the log line arrives a little late)
POST_ROLL_S = 0.25                 # keep recording a moment after release (people let go on the last syllable)
MIN_CLIP_S = 0.35                  # shorter than this = accidental tap
MIN_RMS = 0.003                    # quieter than this (about -50 dBFS) = nothing was said
POLL_WAIT_S = 55
POLL_MIN_INTERVAL_S = 5.05
STATUS_WORTH_S = 3.0
PTT_MAX_S = 30.0                   # a held jump is dropped after this long (the PTT_UP line was lost)
WORLD_MAX_LINE = 100               # the world drops a reply whose line is longer (W04Talk.MaxLine)
LISTEN_MODES = ("gaze", "ptt")     # gaze: open mic while the world says LISTEN_ON; ptt: hold jump only
VAD_FRAME_S = 0.03                 # the detector judges the microphone in 30 ms steps
VAD_ONSET_FRAMES = 8               # speech starts when 5 of the last 8 steps (0.24 s) are loud
VAD_ONSET_VOICED = 5
VAD_END_SILENCE_S = 1.3            # default quiet time before ending the utterance
FRAGMENT_GAP_S = 2.0               # measured from the last voiced frame to the next speech onset
VAD_MAX_S = 15.0                   # an utterance is cut at this length
VAD_MIN_SPEECH_S = 7 * VAD_FRAME_S  # 210 ms of voiced frames; the PTT hold minimum is separate
VAD_TAIL_S = 0.25                  # quiet kept after the last loud step (the rest is trimmed for Whisper)
VAD_MIN_THRESHOLD = 0.0025         # absolute guard when the noise floor is very low
VAD_FLOOR_RATIO = 2.8              # onset must also stand out from the microphone's noise floor
VAD_MIN_CLIP_RMS = 0.0015          # the pre-roll and tail dilute RMS after VAD has already checked the frames
VAD_WEAK_CLIP_RMS = 0.006          # extra Whisper checks for quiet clips and newly allowed short clips
LISTEN_STALE_S = 120               # LISTEN_ON ends by itself if that VRChat log writes nothing this long
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "voice_log.tsv")
CLIP_DIR = os.path.join(HERE, "clips")
VRC_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expandvars(r"%USERPROFILE%\AppData\Local")) + "Low",
                       "VRChat", "VRChat")
SETTINGS_HTML = os.path.join(HERE, "settings_page.html")
SETTINGS_URL_FILE = "settings_url.txt"     # in the store's data folder; settings.cmd opens the address in it
MAX_POST_BYTES = 256 * 1024
WATCH_RECENT_S = 600               # follow every output_log written in the last 10 minutes (several clients)
WATCH_RESCAN_S = 0.5
# VRChat itself writes this line when the local player enters a world: that is who holds the jump button
LOCAL_PLAYER_RE = re.compile(r'Initialized PlayerAPI "(.+)" is local')

# Whisper invents these on silent or noisy clips (its training data had many subtitled videos)
HALLUCINATIONS = re.compile(
    r"^(시청\s*해\s*주셔서\s*감사합니다|구독(과|이랑)?\s*좋아요|좋아요(와|랑)?\s*구독|자막|MBC\s*뉴스|KBS|SBS|"
    r"다음\s*영상에서\s*만나요|오늘도\s*시청)", re.IGNORECASE)

# the world's own events in output_log (CFG_* lines are cfg_bridge's)
TV_STATE_EVENTS = ("TV_STATE", "TV_BACKLIGHT", "TV_SIZE_DONE")
WORLD_STATE_EVENTS = (*TV_STATE_EVENTS, "MIRROR_STATE", "ACTIVITY")
EVENT_RE = re.compile(r"\[W04AI\] (PTT_DOWN|PTT_UP|LISTEN_ON|LISTEN_OFF|NPC_STATE|NPC_CHAT_REQ|CHAT_REQ|AFFECTION_END|AFFECTION|NPC_POSE|GLOBAL|CONTENT_LEVEL|CMD_RESULT|" + "|".join(WORLD_STATE_EVENTS) + r")\b")
# A visitor's text chat sent to this host's app (A2, t20; Tools/contract/memory_v1.json): the world's host
# client logs it, the app answers it like speech with by = the speaker. id (optional) comes back as "req".
CHAT_REQ_RE = re.compile(r"\[W04AI\] CHAT_REQ npc=(kwonjun|itoa)(?: id=([0-9]{1,9}))? by=(.+?) q=(.*?)\s*$")
CHAT_MAX_CHARS = 200
CMD_RESULT_RE = re.compile(r"\[W04AI\] CMD_RESULT npc=(\S+) cmd=([a-z][a-z0-9_]*) ok=([01]) reason=(.+?)\s*$")
CONTENT_LEVEL_RE = re.compile(cfg_bridge._STAMP + r"\[W04AI\] CONTENT_LEVEL value=([0-2])\s*$")
GLOBAL_RE = re.compile(cfg_bridge._STAMP + r"\[W04AI\] GLOBAL on=([01])\s*$")
AFFECTION_RE = re.compile(r"\[W04AI\] AFFECTION type=(pat|hug|wave) npc=(\S+) by=(.+?)\s*$")
AFFECTION_END_RE = re.compile(r"\[W04AI\] AFFECTION_END type=hug npc=(\S+) by=(.+?)\s*$")
POSE_RE = re.compile(r"\[W04AI\] NPC_POSE npc=(\S+) pose=(stand|sit|lie)\b")
NPC_ADDRESS = {
    "kwonjun": re.compile(r"권\s*[준쥰]|(?<![a-z])kwon[\s-]*jung?(?![a-z])", re.IGNORECASE),
    "itoa": re.compile(r"이\s*토\s*[아알]|(?<![a-z])i[\s-]*to[\s-]*ar?(?![a-z])", re.IGNORECASE),
}
NPC_ALIASES = {"itoa": ("이토아", "이토하", "이토와", "토아", "ITOAR", "itoar", "이토")}
SHORT_NPC_ADDRESS = r"(?<!\w){}(?=(?:한테|에게|야|아|가|는)?(?!\w))"
TV_CONTEXT_WORD = re.compile(
    r"(?<![a-z])tv(?![a-z])|티비|티브이|텔레비전|화면|인치|백라이트|밝기|사이즈|크기"
    r"|(?:가로|폭|세로)\s*(?:를|을)?\s*\d+(?:\.\d+)?\s*(?:미터|센티|cm(?![a-z])|m(?![a-z]))",
    re.IGNORECASE,
)
ACTIVITY_RE = re.compile(cfg_bridge._STAMP + r"\[W04AI\] ACTIVITY npc=(\S+) name=(\S+) state=(start|end)\s*$")
ACTIVITY_OBJECT = {"mirror": "거울을", "phone": "폰을", "tablet": "패드를", "laptop": "노트북을"}
MIRROR_STATE_RE = re.compile(cfg_bridge._STAMP + r"\[W04AI\] MIRROR_STATE npc=(\S+) players=(.+?)\s*$")
MIRROR_WORD = re.compile(r"거울|미러|mirror|鏡", re.IGNORECASE)
CMD_RESULT_TTL_S = 60
CMD_RESULT_REASONS = {
    "done": "성공", "blocked": "길 막힘", "timeout": "시간 초과", "busy": "다른 사람이 쓰는 중",
    "no_permission": "권한 없음", "out_of_range": "범위 밖", "left": "상대가 나감",
    "dispatched": "요청 보냄", "delegated": "다른 화면에서 처리", "already_done": "이미 그 상태",
    "invalid_argument": "잘못된 값", "owner_left": "처리하던 사람이 나감", "unavailable": "쓸 수 없음",
    "unsupported": "지원 안 됨", "no_video": "재생 중인 영상 없음", "loading": "영상 불러오는 중",
    "ended": "끝남", "path_changed": "길이 바뀜",
}
# Neither a success nor a failure: replaced by another command, stopped on request, handed to a new
# owner, or not known. The AI is never told about these.
CMD_RESULT_NEUTRAL = frozenset(("superseded", "stopped", "unknown", "owner_changed", "unconfirmed"))
CMD_RESULT_QUESTION = re.compile(r"됐어|됐니|됐나|됐냐|했(?:어|니|나|냐)\s*[?？]|어떻게\s*됐|왜\s*안|왜\s*못|안\s*됐|못\s*했")
CMD_REQUEST_WORD = re.compile(
    r"해\s*줘|주세요|해\s*봐|(?:켜|꺼|끄자|당겨|밀어|올려|내려|돌려|초기화|재생|멈춰|정지|키워|줄여|낮춰|밝혀|"
    r"가|와|앉아|누워|일어나|일어서|엎드려|벗어|입어|놔|풀어|놓아|바꿔|맞춰|설정해|좁혀|떨어져|가자|최대|최소)"
    r"\s*(?:줘|줘요|봐|주라|줄래|주세요)?\s*(?:[.!?。！？]|$)"
    r"|\b(?:turn|switch|set|play|stop|pause|move|follow|come|sit|stand|take off|put on|release)\b", re.IGNORECASE)
CMD_KIND_WORDS = (
    (persona_ai.TV_ACTIONS, TV_CONTEXT_WORD),
    (("move_to", "come", "follow", "stop", "keep_distance", "go_sit"),
     re.compile(r"이리|여기|저기|이쪽|저쪽|가까이|멀리|거리|따라|멈춰|창가|걸어|(?:가|와)\s*(?:줘|봐|[.!?]|$)")),
    (persona_ai.POSE_ACTIONS + ["go_sit"], re.compile(r"앉|눕|누워|일어나|일어서|서\s*있|엎드")),
    (("outfit",), re.compile(r"옷|신발|양말|넥타이|목리본|벗어|입어")),
    (("mirror",), MIRROR_WORD),
    (("night_mode",), re.compile(r"나이트|야간|밤\s*모드|night\s*mode", re.IGNORECASE)),
    (("hug_release",), re.compile(r"안기|포옹|놔\s*줘|놓아\s*줘|hug", re.IGNORECASE)),
)
FOLLOW_UP_MS = 20_000
DEFAULT_NPC_STATE = {"kwonjun": True, "itoa": False}


TOUCH_NOW_S = 15
TOUCH_HUG_MAX_S = 90
touch_now = {}
touch_lock = threading.Lock()


def npc_called(text, npc, display_name):
    """The words call that NPC ('이토아, 이리 와'), not only mention it ('이토아는 뭐 좋아해?')."""
    return smalltalk.called(text, (*smalltalk.REACTION_NAMES[npc], *NPC_ALIASES.get(npc, ()), display_name))


def npc_named(text, npc, display_name):
    if NPC_ADDRESS[npc].search(text):
        return True
    names = list(NPC_ALIASES.get(npc, ()))
    if display_name and display_name != persona_store.DEFAULT_NPC_NAMES[npc]:
        names.append(display_name)
    for name in names:
        pattern = SHORT_NPC_ADDRESS if name in ("토아", "이토") else r"(?<![a-z]){}(?![a-z])"
        if re.search(pattern.format(re.escape(name)), text, re.IGNORECASE):
            return True
    return False


def update_tv_state(state, event, line):
    values = {key: int(value) for key, value in
              re.findall(r"\b(inches|min|max|backlight|percent)=(-?\d+)(?=\s|$)", line)}
    if event == "TV_STATE":
        values.pop("percent", None)
        for key, value in values.items():
            if value < 0:
                state.pop(key, None)
    elif event == "TV_SIZE_DONE" and re.search(r"\bfits=true\b", line, re.IGNORECASE):
        values = {"inches": values.get("inches", -1)}
    elif event == "TV_BACKLIGHT":
        values = {"backlight": values.get("percent", -1)}
    else:
        return
    state.update((key, value) for key, value in values.items() if value >= 0)
    state["at"] = time.monotonic()


def update_world_state(tail, event, line):
    if event in TV_STATE_EVENTS:
        update_tv_state(tail.tv_state, event, line)
        return
    if event == "ACTIVITY":
        match = ACTIVITY_RE.match(line)
        if not match or match[1] not in DEFAULT_NPC_STATE or match[2] not in ACTIVITY_OBJECT:
            return
        npc, name, phase = match.groups()
        try:
            at = datetime.strptime(line[:19], "%Y.%m.%d %H:%M:%S").timestamp()
        except (ValueError, OSError, OverflowError):
            return
        state = tail.activity_state.setdefault(npc, {})
        if phase == "start":
            state["current"], state["started_at"] = name, at
        else:
            if state.get("current") == name:
                del state["current"]
            state["last"], state["ended_at"] = name, at
        return
    match = MIRROR_STATE_RE.match(line)
    if not match or match[1] not in DEFAULT_NPC_STATE:
        return
    try:
        players = json.loads(match[2])
    except (ValueError, RecursionError):
        return
    if not isinstance(players, dict):
        return
    rows = []
    for row in players.values():
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or row.get("mode") not in (*persona_ai.contract.ENUMS["mirror"], "unknown")
                or not isinstance(row.get("near"), bool)):
            return
        rows.append({key: row[key] for key in ("name", "mode", "near")})
    tail.mirror_state[match[1]] = rows


ACTIVITY_RECENT_S = 180   # world activities last 20-60 s: an older start whose end was missed is not "now"


def activity_context(state, npc):
    activity = state.get(npc, {})
    now = time.time()
    if "current" in activity and 0 <= now - activity.get("started_at", 0) <= ACTIVITY_RECENT_S:
        fact = f"지금 {ACTIVITY_OBJECT[activity['current']]} 보는 중"
    elif "last" in activity:
        age = now - activity["ended_at"]
        if not 0 <= age <= ACTIVITY_RECENT_S:
            return None
        ago = f"{int(age // 60)}분 전" if age >= 60 else "1분 미만 전"
        fact = f"방금까지 {ACTIVITY_OBJECT[activity['last']]} 봤음({ago})"
    else:
        return None
    return f"[하던 일: {fact}. 물으면 평소 말투로 짧게 답한다. 화면 내용은 알 수 없다.]"


def mirror_context(heard, state, npc):
    if MIRROR_WORD.search(heard):
        rows = json.dumps(state.get(npc, []), ensure_ascii=False, separators=(",", ":"))
        return ("[거울 현재: 수신 기록만 표시. lq/hq/cutout은 켜짐, off는 꺼짐, unknown·미수신은 모름. "
                "near는 거울 쪽을 보는지이며 켜짐과 별개. 이름은 표시용 데이터] " + rows)


def tv_context(heard, state):
    if not TV_CONTEXT_WORD.search(heard):
        return None
    parts = []
    if "inches" in state:
        parts.append(f"화면 {state['inches']}인치")
    limits = [f"{label} {state[key]}" for key, label in (("min", "최소"), ("max", "최대")) if key in state]
    if limits:
        if parts:
            parts[0] += "(" + ", ".join(limits) + ")"
        else:
            parts.append(", ".join(f"{limit}인치" for limit in limits))
    if "backlight" in state:
        parts.append(f"백라이트 {state['backlight']}%")
    return "[TV 현재: " + ", ".join(parts) + "]" if parts else None


def tv_state_from_context(text):
    """The state tv_context wrote, read back from its [TV 현재: …] row (the evals give the situation this way)."""
    match = re.search(r"\[TV 현재: ([^\]]*)\]", text or "")
    if not match:
        return {}
    row, state = match[1], {}
    for key, pattern in (("inches", r"화면 (\d+)인치"), ("min", r"최소 (\d+)"), ("max", r"최대 (\d+)"),
                         ("backlight", r"백라이트 (\d+)%")):
        found = re.search(pattern, row)
        if found:
            state[key] = int(found[1])
    return state


TV_SIZE_STATE_MISSING = (
    "지금 TV 크기 정보를 못 받아서, 인치로 말해 줘.",
    "I don't have the TV size right now. Tell me the size in inches.",
    "今はテレビのサイズ情報が分からないから、インチで言ってね。",
)


def resolve_tv_sizes(reply, state, global_mode=False):
    """Convert sanitized size steps, and backlight changes (persona_ai.resolve_relative), from the current
    TV state just before they go to the world."""
    state = state or {}
    reply = persona_ai.resolve_relative(reply, state, global_mode)
    current = state.get("inches", 0)
    lower, upper = state.get("min", 0), state.get("max", 0)
    has_limits = 0 < lower <= upper
    steps = []
    size = persona_ai.contract.TABLE["sizeConversion"]
    cap = persona_ai.contract.ARGS["tv_size_set"]["fields"]["inches"]["max"]
    increment = size["increment"]
    missing = False
    for step in reply["steps"]:
        if step["do"] != "tv_size_set" or "inches" in step:
            steps.append(step)
            continue
        if "percent" in step or "percent_delta" in step:
            if not has_limits or current <= 0:
                missing = True   # only this step is left out (the table's rule); the others still run
                continue
            if "percent" in step:
                inches = lower + (upper - lower) * step["percent"] / 100
            else:
                inches = current * (100 + step["percent_delta"]) / 100
        else:
            key, ratio = ("width_m", size["widthRatio"]) if "width_m" in step else ("height_m", size["heightRatio"])
            inches = min(step[key], cap) * size["inchesPerMeter"] / ratio
        # Match the world's midpoint rounding and keep the outgoing value positive.
        inches = max(increment, round(min(inches, cap) / increment) * increment)
        if has_limits:
            inches = max(lower, min(inches, upper))
        steps.append({"do": "tv_size_set", "target": "none", "gesture": "none", "inches": inches})
    reply = dict(reply, steps=steps)
    if missing:
        if all(step["do"] in persona_ai.DECORATIVE_ACTIONS for step in steps):
            return dict(reply, **persona_ai.command_refusal(TV_SIZE_STATE_MISSING, global_mode))
        # Other steps run: the line keeps them and says why the size did not change (a line naming the
        # size % it promised is replaced, like persona_ai.PERCENT_NOTES).
        note = persona_ai.localized_lines(TV_SIZE_STATE_MISSING)
        lines = {lang: persona_ai.join_note(value, note[lang], bool(persona_ai.PERCENT_MENTION.search(value)))
                 for lang, value in reply["lines"].items()}
        reply.update(lines=lines, line=lines["ko"])
    return reply


def update_command_results(results, line):
    match = CMD_RESULT_RE.search(line)
    if not match or match[1] not in DEFAULT_NPC_STATE or match[4] in CMD_RESULT_NEUTRAL:
        return
    results.setdefault(match[1], collections.deque(maxlen=3)).append(
        {"at": time.monotonic(), "cmd": match[2], "ok": match[3] == "1", "reason": match[4], "told": False})


def command_result_context(heard, results, npc):
    """What just happened in the world, for this turn. The world log does not say who gave the command,
    so the line is neutral. A failure is told once; after that (and any success) only when the player
    asks how it went ("됐어?", "왜 안 됐어?") or talks about the same kind of command, within 60 s.
    Call with the lock that guards `results` (the told marks are updated here)."""
    now = time.monotonic()
    recent = [item for item in results.get(npc, ()) if now - item["at"] < CMD_RESULT_TTL_S]
    if not recent:
        return None
    commands = {item["cmd"] for item in recent}
    asked = bool(CMD_RESULT_QUESTION.search(heard))
    matches_kind = CMD_REQUEST_WORD.search(heard) and any(
        commands.intersection(kind) and pattern.search(heard) for kind, pattern in CMD_KIND_WORDS)
    matches_name = any(re.search(rf"(?<![a-z0-9_]){cmd}(?![a-z0-9_])", heard, re.IGNORECASE) for cmd in commands)
    related = bool(matches_kind or matches_name)
    chosen = [item for item in recent
              if asked or (not item["told"] and (not item["ok"] or related))]
    if not chosen:
        return None
    parts = []
    for item in chosen:
        item["told"] = True
        status = "성공" if item["ok"] else "실패"
        label = CMD_RESULT_REASONS.get(item["reason"])
        if item["reason"] != "done" and label:
            status += f"({label})"
        parts.append(f"{item['cmd']} {status}")
    return "[방금 월드에서 일어난 일: " + " / ".join(parts) + "]"


def clear_touch(path=None):
    """Forget the hug/pat situation that came from this log file (all of them when path is None)."""
    with touch_lock:
        for npc in [npc for npc, item in touch_now.items() if path is None or item.get("path") in (None, path)]:
            del touch_now[npc]


def touch_context(npc):
    with touch_lock:
        item = touch_now.get(npc)
        if item is None:
            return None
        now = time.monotonic()
        who = item["by"]
        if item["type"] == "hug":
            ended = item["ended"]
            if ended is None and now - item["at"] <= TOUCH_HUG_MAX_S:
                return f"[지금: {who}가 너를 안고 있음]"
            if ended is not None and now - ended <= TOUCH_NOW_S:
                return f"[방금: {who}가 너를 안았다가 놓음]"
        elif now - item["at"] <= TOUCH_NOW_S:
            return f"[방금: {who}가 네 머리를 쓰다듬음]"
        del touch_now[npc]
        return None


def end_silence_from_env():
    try:
        seconds = float(os.environ.get("W04_END_SILENCE_S", VAD_END_SILENCE_S))
    except ValueError:
        return VAD_END_SILENCE_S
    return seconds if math.isfinite(seconds) and 0 < seconds < VAD_MAX_S else VAD_END_SILENCE_S

def event_npc(line):
    m = re.search(r"(?:^|\s)npc=([^\s]+)", line)
    return persona_store.npc_id(m.group(1) if m else None)


def update_npc_state(state, line):
    if re.search(r"\[W04AI\] NPC_STATE\b", line):
        for npc, enabled in re.findall(r"\b(kwonjun|itoa)=([01])\b", line):
            state[npc] = enabled == "1"


class NpcBrains:
    """One lazily created brain (and fixed prompt cache) per NPC."""
    def __init__(self, mode, store):
        self.mode, self.store = mode, store
        self.brains = {}
        self.init_ms = {}
        self.lock = threading.RLock()

    def for_npc(self, npc=None):
        npc = persona_store.npc_id(npc)
        with self.lock:
            if npc not in self.brains:
                started = time.monotonic()
                self.brains[npc] = make_brain(self.mode, self.store.for_npc(npc))
                self.init_ms[npc] = round((time.monotonic() - started) * 1000)
            return self.brains[npc]

    def take_init_ms(self, npc):
        with self.lock:
            return self.init_ms.pop(npc, 0)


def start_brain_warmup(brain):
    if brain.mode not in MODELS or not os.environ.get("ANTHROPIC_API_KEY"):
        return

    def warm():
        for npc in ("kwonjun", "itoa"):
            started = time.monotonic()
            try:
                with brain.lock:
                    ready = brain.for_npc(npc)
                    brain.take_init_ms(npc)
                    if ready is None:
                        # make_brain already explained the missing SDK. Leave the slot retryable.
                        del brain.brains[npc]
                        continue
            except Exception:
                say_line(f"{npc} 대화 준비에 실패했어요. 첫 말에 다시 시도합니다.")
                continue
            log("BRAIN_READY", f"npc={npc}\tms={elapsed_ms(started, time.monotonic())}")

    thread = threading.Thread(target=warm, daemon=True)
    thread.start()
    return thread


def select_brain(brain, npc):
    if isinstance(brain, NpcBrains):
        return brain.for_npc(npc)
    if npc == "kwonjun" or brain is None:
        return brain
    # Compatibility for callers passing an existing Brain.
    if not hasattr(brain, "npc_brains"):
        brain.npc_brains = NpcBrains(getattr(brain, "mode", "off"), brain.store)
    return brain.npc_brains.for_npc(npc)


HEARD_FMT = "들은 말: {text}"
MISSED_LINE = "응? 잘 못 들었어."

_print_lock = threading.Lock()
_file_lock = threading.Lock()
_log_missed = 0


def now_ms() -> float:
    return time.time() * 1000.0


def ai_error_hint(err: str) -> str:
    """A plain Korean next step for the AI errors a first-time user can fix alone."""
    if "AuthenticationError" in err or "invalid x-api-key" in err:
        return ("\n  → API 키가 틀렸습니다. setup.cmd를 다시 실행해 키를 새로 붙여 넣거나, .env의 ANTHROPIC_API_KEY= 줄을 "
                "확인하세요(앞뒤 띄어쓰기·따옴표 없이 sk-ant-로 시작하는 키 전체). 고친 뒤 이 창을 닫고 다시 켜세요.")
    if "credit balance" in err or "billing" in err.lower():
        return "\n  → API 크레딧이 없습니다. console.anthropic.com의 Billing에서 충전한 뒤 다시 말해 보세요."
    if "PermissionDenied" in err:
        return "\n  → 이 키로는 쓸 수 없는 모델입니다. Console에서 키의 권한(워크스페이스)을 확인하세요."
    return ""


def translation_flag(lines, lang) -> str:
    value = lines.get(lang)
    if not value:
        return "0"
    return "fallback" if value == persona_ai.UNAVAILABLE_TRANSLATION.get(lang) else "1"


def say_line(message):
    with _print_lock:
        try:
            print(time.strftime("%H:%M:%S ") + message, flush=True)
        except (OSError, ValueError):
            pass


def log(kind: str, detail: str = "", say: str = "") -> None:
    """One line to voice_log.tsv; `say` (Korean) is what the user sees in the console."""
    global _log_missed
    stamp = f"{now_ms():.0f}"
    with _file_lock:
        try:
            with open(LOG_PATH, "a", encoding="utf-8", errors="backslashreplace") as f:
                recovery = f"{stamp}\tLOG_RECOVERED\tmissed={_log_missed}\n" if _log_missed else ""
                f.write(f"{recovery}{stamp}\t{kind}\t{detail}\n")
        except OSError:
            _log_missed += 1
            if _log_missed == 1:
                say_line("기록 파일을 쓸 수 없어요(다른 프로그램에서 열려 있나요?). 대화는 계속됩니다.")
        else:
            _log_missed = 0
    if say:
        say_line(say)


def exception_location(error):
    frame = traceback.extract_tb(error.__traceback__, limit=-1)[0]
    return f"{type(error).__name__}\t{PurePath(frame.filename).name}:{frame.lineno}"


def supervise_thread(name, target, stop=None):
    failures = collections.deque()
    while stop is None or not stop.is_set():
        try:
            target()
            return
        except Exception as error:
            now = time.monotonic()
            while failures and now - failures[0] > 60:
                failures.popleft()
            failures.append(now)
            restart = len(failures) < 5
            log("THREAD_ERROR", f"name={name}\t{exception_location(error)}",
                say=f"{name} 작업 중 오류가 났어요. " +
                    ("1초 뒤 다시 시작합니다." if restart else "앱을 다시 켜 주세요."))
            if not restart:
                return
        if stop is None:
            time.sleep(1)
        elif stop.wait(1):
            return


def start_compaction(st, brain, force: bool = False, wait: bool = False) -> str:
    """Memory compaction for one NPC on its own thread (called AFTER the reply was queued, so a reply
    never waits for it). Returns '' when a run started, else why not. One short log line per run."""
    if brain is None or not hasattr(brain, "compact_memory"):
        return "AI가 꺼져 있어서 정리할 수 없습니다."
    job, why = st.compaction_begin(force)
    if job is None:
        return why

    def run():
        try:
            result, usage = brain.compact_memory(job)
            done = st.compaction_finish(job, result, usage)
        except Exception as e:  # network, refusal, cut-off answer: nothing was changed, try again later
            done = st.compaction_finish(job, error=type(e).__name__)
        b, a = done["notes_before"], done["notes_after"]
        people = done["impressions"]
        use = done["usage"] or {"in": 0, "out": 0}
        detail = (f"npc={st.npc}\tok={int(done['ok'])}\tnotes={b['count']}>{a['count']}\tchars={b['chars']}>{a['chars']}"
                  f"\timpressions={len(people)}\tin={use['in']}\tout={use['out']}\tusd={done['cost_usd']:.6f}"
                  f"\t{done['error']}")
        if done["ok"]:
            say = (f"  └ {st.npc_name()}의 기억을 정리함: 메모 {b['count']}개 {b['chars']:,}자 → "
                   f"{a['count']}개 {a['chars']:,}자" + (f", 인상 {len(people)}명" if people else "")
                   + " (설정 화면에서 되돌릴 수 있음)")
        else:
            say = f"  └ {st.npc_name()}의 기억 정리는 바뀐 것 없이 끝남 ({done['error']})"
        log("COMPACT", detail, say=say)

    if wait:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()
    return ""


def start_memory(st, brain, name: str, force: bool = False, wait: bool = False) -> str:
    """After a reply went out ("대화 뒤 기억 정리", t20): the memory call for that person -- profile,
    impression, new memos -- then the compaction check, on one thread, so a reply never waits for either.
    While a call for that person runs, ONE more run is queued; one person's runs start MEMORY_MIN_GAP_S
    apart unless force ("기억해 줘"). A failure changes nothing and never reaches the reply: log line
    MEMORY_CALL ok=0 and an error row (A122) in the activity record. Returns '' when a run started."""
    if brain is None or not hasattr(brain, "update_memory"):
        return start_compaction(st, brain, wait=wait)
    delay = st.memory_claim(name, force)
    if delay is None:               # "auto_update" off, or queued behind the running call for that person
        return start_compaction(st, brain, wait=wait)

    def run():
        nonlocal delay
        while delay is not None:
            if delay > 0:
                time.sleep(delay)
            try:
                job = st.memory_begin(name)
                if job is not None:
                    started = time.monotonic()
                    try:
                        result, usage = brain.update_memory(job)
                        done = st.memory_finish(job, result, usage)
                    except Exception as e:  # network, refusal, cut-off answer: nothing was changed
                        done = st.memory_finish(job, error=type(e).__name__)
                    report_memory(st, done, round((time.monotonic() - started) * 1000))
            except Exception as e:          # never leave the claim behind: the next reply must start one again
                log("MEMORY_CALL", f"npc={st.npc}\tok=0\t{type(e).__name__}\t{activity.log_code('A122')}")
            delay = st.memory_release(name)
        start_compaction(st, brain, wait=True)

    if wait:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()
    return ""


def report_memory(st, done: dict, ms: int):
    """Log lines (MEMORY_CALL, then IMPRESSION / PROFILE / MEMORY with their text) and the activity rows of
    one memory call. The rows reach /cfg at once and the next reply's act (activity.Journal.later)."""
    use = done["usage"] or {"in": 0, "out": 0}
    name, npc = done["name"], st.npc
    changed = bool(done["impression"] or done["profile"] or done["notes"])
    code = "" if done["ok"] else "A122"
    log("MEMORY_CALL", f"npc={npc}\twho={name}\tok={int(done['ok'])}\tchanged={int(changed)}\tms={ms}\t"
                       f"in={use['in']}\tout={use['out']}\tusd={done['cost_usd']:.6f}\t{done['error']}"
                       + (f"\t{activity.log_code(code)}" if code else ""),
        say=f"  └ {st.npc_name()}: 대화 뒤 기억 정리 실패 ({done['error']}, {code})" if code else "")
    if code:
        activity.journal.later(npc, "error", activity.code_text(code), code=code)
        return
    label = st.player_label(name)
    if done["impression"]:
        before, after = done["impression"]["before"], done["impression"]["after"]
        log("IMPRESSION", f"npc={npc}\t{name}\t{after}",
            say=f"  └ {label}에 대한 인상을 고쳐 씀: {before or '(없음)'} → {after}")
        activity.journal.later(npc, "impression", f"{label}: {after}")
    if done["profile"]:
        log("PROFILE", f"npc={npc}\t{name}\t{' / '.join(done['profile'])}",
            say=f"  └ {label}에 대해 아는 것: {' / '.join(done['profile'])}")
        if not done["impression"]:
            activity.journal.later(npc, "impression", f"{label}: {' / '.join(done['profile'])}")
    for note in done["notes"]:
        log("MEMORY", f"npc={npc}\t{note}", say=f"  └ 기억에 기록함: {note}")
        activity.journal.later(npc, "memory", note)


# ---------------------------------------------------------------- microphone

class Recorder:
    """Keeps the mic open; a short ring buffer covers the moment before the press."""

    mute = None

    def __init__(self, device=None, mute=None):
        import sounddevice as sd
        self.sd = sd
        self.device = device
        self.lock = threading.Lock()
        self.mute = mute
        self.mute_epoch = 0
        self.recording = False
        self.chunks = []
        self.tap = None                 # gaze mode: a queue that gets every block while it is open
        info = sd.query_devices(device, "input")
        self.name = info["name"]
        self.rate = RATE
        try:
            sd.check_input_settings(device=device, samplerate=RATE, channels=1, dtype="float32")
        except Exception:
            self.rate = int(info["default_samplerate"])   # resampled to 16 kHz later
        self.pre = collections.deque(maxlen=max(1, int(PRE_ROLL_S * self.rate / 512) + 1))
        self.max_chunks = int((PTT_MAX_S + 5.0) * self.rate / 512)
        self.stream = sd.InputStream(device=device, samplerate=self.rate, channels=1, dtype="float32",
                                     blocksize=512, callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, t, status):
        with self.lock:
            if self.mute is not None:
                muted, epoch = self.mute.snapshot
                if epoch != self.mute_epoch:
                    self.mute_epoch = epoch
                    self.pre.clear()
                    self.chunks.clear()
                if muted:
                    return
            block = indata[:, 0].copy()
            if self.tap is not None:
                self.tap.put(block)
            if self.recording and len(self.chunks) < self.max_chunks:
                self.chunks.append(block)
            elif self.recording:
                pass                    # held far too long: keep what there is, take no more
            else:
                self.pre.append(block)

    def open_tap(self) -> "queue.Queue":
        """Gaze mode: from now on every block also goes into the returned queue, starting with the
        0.5 s already in the ring. Nothing reaches the listener before this call or after close_tap."""
        with self.lock:
            self.tap = queue.Queue()
            if self.mute is None or self.mute.snapshot == (False, self.mute_epoch):
                for block in self.pre:
                    self.tap.put(block)
            return self.tap

    def close_tap(self):
        with self.lock:
            self.tap = None

    def start(self):
        with self.lock:
            self.chunks = (list(self.pre) if self.mute is None or
                           self.mute.snapshot == (False, self.mute_epoch) else [])
            self.pre.clear()
            self.recording = True

    def stop(self) -> np.ndarray:
        time.sleep(POST_ROLL_S)
        with self.lock:
            self.recording = False
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
            self.chunks = []
        return to_16k(audio, self.rate)

    def cancel(self):
        """Stop recording now and throw the audio away (world change, lost release)."""
        with self.lock:
            self.recording = False
            self.chunks = []


class WavRecorder:
    """Test stand-in for the mic: every press 'records' the same WAV file."""

    rate = RATE

    def __init__(self, path):
        self.audio = read_wav(path)
        self.name = "WAV " + os.path.basename(path)

    def open_tap(self) -> "queue.Queue":
        """Gaze mode: every LISTEN_ON 'hears' the WAV once, then silence (all at once, not in real time)."""
        tap = queue.Queue()
        audio = np.concatenate([self.audio, np.zeros(int((end_silence_from_env() + 0.3) * RATE), np.float32)])
        for i in range(0, audio.size, 512):
            tap.put(audio[i:i + 512])
        return tap

    def close_tap(self):
        pass

    def start(self):
        pass

    def stop(self) -> np.ndarray:
        return self.audio


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE or audio.size == 0:
        return audio.astype(np.float32)
    n = int(round(audio.size * RATE / rate))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit("WAV must be 16-bit PCM")
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return to_16k(a, rate)


def save_clip(audio: np.ndarray, seq: int) -> str:
    os.makedirs(CLIP_DIR, exist_ok=True)
    path = os.path.join(CLIP_DIR, time.strftime("%Y%m%d_%H%M%S") + f"_{seq}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


class UtteranceDetector:
    """Voice activity detection for gaze mode: an energy gate with an adaptive noise floor.

    feed(block) takes microphone blocks of any size and returns the utterances that ended inside that
    block as (audio at the microphone's rate, seconds of loud time). The caller drops those under
    VAD_MIN_SPEECH_S. Everything is counted in samples, so a test may feed faster than real time.

    The floor follows quiet steps quickly and loud ones slowly, so a steady noise (fan, music) stops
    counting as speech after a while but the gaps between words keep it down while someone talks.
    It cannot tell a voice from other sudden sounds; Whisper's own checks (judge) are the second net."""

    def __init__(self, rate: int):
        self.rate = rate
        self.n = max(1, int(rate * VAD_FRAME_S))
        self.end_silence = end_silence_from_env()
        self.floor = 0.002
        self.reset()

    def reset(self):
        """Forget all audio (the noise floor stays)."""
        self.rest = np.zeros(0, np.float32)
        self.pre = collections.deque(maxlen=int(PRE_ROLL_S / VAD_FRAME_S) + 1)
        self.recent = collections.deque(maxlen=VAD_ONSET_FRAMES)
        self.frames = []
        self.active = False             # inside an utterance
        self.voiced = 0
        self.quiet = 0

    def feed(self, block: np.ndarray) -> list:
        out = []
        block = np.asarray(block, np.float32)
        data = np.concatenate([self.rest, block]) if self.rest.size else block
        end = data.size - data.size % self.n
        for i in range(0, end, self.n):
            done = self._frame(data[i:i + self.n])
            if done is not None:
                out.append(done)
        self.rest = data[end:]
        return out

    @property
    def onset_threshold(self):
        return max(VAD_MIN_THRESHOLD, self.floor * VAD_FLOOR_RATIO)

    def _frame(self, f: np.ndarray):
        rms = float(np.sqrt(np.mean(f * f)))
        up = 0.001 if self.active else 0.003
        self.floor = max(0.0005, self.floor + (rms - self.floor) * (0.2 if rms < self.floor else up))
        threshold = self.onset_threshold
        if not self.active:
            self.pre.append(f)
            self.recent.append(rms >= threshold)
            if sum(self.recent) >= VAD_ONSET_VOICED:
                self.active = True
                self.frames = list(self.pre)        # includes the pre-roll and the onset itself
                self.voiced, self.quiet = sum(self.recent), 0
                self.pre.clear()
                self.recent.clear()
            return None
        self.frames.append(f)
        if rms >= threshold * 0.6:                  # easier to stay in than to get in
            self.voiced += 1
            self.quiet = 0
        else:
            self.quiet += 1
        if self.quiet * VAD_FRAME_S < self.end_silence and len(self.frames) * VAD_FRAME_S < VAD_MAX_S:
            return None
        self.trailing_silence = self.quiet * VAD_FRAME_S
        cut = max(0, self.quiet - int(VAD_TAIL_S / VAD_FRAME_S))
        audio = np.concatenate(self.frames[:len(self.frames) - cut])
        speech_s = self.voiced * VAD_FRAME_S
        self.frames, self.active, self.voiced, self.quiet = [], False, 0, 0
        return audio, speech_s


# ---------------------------------------------------------------- speech to text

# Collapse long runs of a non-numeric character or a short repeated syllable group.
TRANSCRIPT_RUN = re.compile(r"([^\s\d])\1{5,}|([가-힣ㄱ-ㅎㅏ-ㅣA-Za-z]{1,4}?)(?: ?\2){5,}")
TERMINAL_PUNCT = re.compile(r'''[.!?。！？…]["'”’」』)]*$''')
SENTENCE_END = re.compile(
    r"(?:니다|니까|어요|아요|세요|해요|예요|네요|군요|겠다|한다|했다|이다|더라|잖아|줄래|할래|거야|구나|했어)$")
CASUAL_END = re.compile(r"(?:줘|봐|해|돼|와|자|어|아|야|지|네|래|까|요|다|죠|군|걸|게|마|라|렴|니|냐|나"
                        r"|꺼|켜|워|파|퍼|빠|려|겨|져|쳐|떠|써|짜|박)$")
# "창가로 가": a spaced 가 is the verb (go), not the subject particle of "네가".
VERB_GA_END = re.compile(r"\s가$")
# "그래서 어떻게 됐는데": a question word makes a 는데 ending a question.
QUESTION_NDE_END = re.compile(r"(?:뭐|무엇|어떻게|왜|언제|어디|누가|누구|몇|얼마).*(?:는데|은데|ㄴ데|인데)$")
# "1미터", "50%": a bare amount answers a question.
AMOUNT_END = re.compile(r"\d\s*(?:미터|m|센티|cm|인치|초|분|시간|퍼센트|%|번|개|도)$", re.IGNORECASE)
UNFINISHED_END = re.compile(
    r"(?P<word>(?:^|\s)(?:그리고|근데|그런데|그래서|그러니까|그럼|아니|음|어|저기|있잖아))$"
    r"|(?:고|는데|은데|ㄴ데|인데|서|면|지만|니까|랑|이랑|하고|거나|든지|은|는|이|가|을|를|에|에서|로|으로|와|과|도|만|의)$")
STT_FILLERS = re.compile(r"(?:[아어으음흠허헉앗와오하흐헤에엥헐ㅋㅎㅠㅜ]+|uh|um|hmm|ah|oh)", re.IGNORECASE)
STT_SHORT_WORDS = frozenset(
    ("네 예 응 넵 아니 안돼 맞아 안녕 좋아 싫어 됐어 그래 잠깐 "
     "앉아 멈춰 거울 권준 켜 꺼 가 줘 해 왜 뭐 뭐해 누가 언제 어디 지금 TV OK NO "
     "그만 미안 진짜 정말 대박 오케이 가까이 멀리 고마워 반가워 배고파 최고 대단해 굿 땡큐").casefold().split()
)


def collapse_transcript_runs(text):
    return TRANSCRIPT_RUN.sub(lambda m: (m[1] or m[2]) * (3 if m[1] or len(m[2]) == 1 else 2), text)


def is_fragment(text):
    text = text.strip()
    if TERMINAL_PUNCT.search(text):
        return False
    text = text.rstrip(" ,，、:;\"'”’」』)")
    if text.casefold() in STT_SHORT_WORDS:
        return False
    unfinished = UNFINISHED_END.search(text)
    # A whole filler or connector can look like a casual ending (어, 있잖아).
    if unfinished and unfinished["word"] is not None:
        return True
    if (SENTENCE_END.search(text) or CASUAL_END.search(text) or VERB_GA_END.search(text)
            or QUESTION_NDE_END.search(text) or AMOUNT_END.search(text)):
        return False
    return len("".join(text.split())) < 4 or unfinished is not None

class Transcriber:
    supports_initial_prompt = True
    beam_size = None

    def __init__(self, model_name: str):
        import torch
        import whisper
        self.whisper = whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beam_size = whisper_beam_size(os.environ.get("W04_WHISPER_BEAM_SIZE"))
        # model folder: WHISPER_MODEL_DIR in .env, else Whisper's own default (%USERPROFILE%\.cache\whisper)
        root = os.environ.get("WHISPER_MODEL_DIR") or os.path.join(
            os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
        url = whisper._MODELS.get(model_name, "")
        if url and not os.path.exists(os.path.join(root, os.path.basename(url))):
            print(f"처음 한 번 Whisper 모델 '{model_name}'을 받습니다(turbo는 약 1.6 GB). 받는 곳: {root}", flush=True)
        if self.device == "cpu":
            print("※ NVIDIA 그래픽카드(CUDA)를 못 찾아서 CPU로 계산합니다. 동작은 하지만 글자 변환이 몇 초 걸릴 수 있습니다.", flush=True)
        t0 = now_ms()
        self.model = whisper.load_model(model_name, device=self.device, download_root=root)
        try:
            self.run(np.zeros(RATE, np.float32))      # warm-up: the first call is slow
        except RuntimeError as e:                     # e.g. a GPU too old for this PyTorch build
            if self.device != "cuda":
                raise
            print(f"※ 그래픽카드로 계산하다 오류가 나서 CPU로 바꿉니다: {str(e)[:120]}", flush=True)
            self.device = "cpu"
            self.model = whisper.load_model(model_name, device="cpu", download_root=root)
            self.run(np.zeros(RATE, np.float32))
        self.load_ms = now_ms() - t0

    def run(self, audio: np.ndarray, initial_prompt: str = None):
        decode_options = {"beam_size": self.beam_size} if self.beam_size else {}
        result = self.model.transcribe(
            audio, language="ko", task="transcribe", fp16=(self.device == "cuda"),
            temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
            no_speech_threshold=0.6, logprob_threshold=-1.0, initial_prompt=initial_prompt, **decode_options)
        return result


def transcript_key(text):
    return "".join(char for char in text.casefold() if char.isalnum())


def is_unclear_transcript(text, pairs):
    word = transcript_key(text)
    if word in STT_SHORT_WORDS or word.isdecimal() or any(word == transcript_key(target) for _, target in pairs):
        return False
    return not word or len(word) <= 2 or STT_FILLERS.fullmatch(word) is not None


def checked_transcript(result, require_clear_speech):
    segs = result.get("segments", [])
    if segs and all(s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0 for s in segs):
        return "", "no_speech"
    text = collapse_transcript_runs(" ".join(result.get("text", "").split()))
    if not text:
        return "", "empty"
    if require_clear_speech and not any(
        s.get("no_speech_prob", 1) < 0.5 and s.get("avg_logprob", -math.inf) >= -0.75 for s in segs
    ):
        return "", "weak_speech"
    if HALLUCINATIONS.search(text):
        return "", f"hallucination '{text}'"
    return text, ""


def judge(audio: np.ndarray, held_ms: float, tr: Transcriber, hints: str = "", raw_sink=None,
          vad_speech_s: float | None = None):
    """Returns (text or '', reason, stt_ms). held_ms = how long jump was held (from the world's log line)."""
    dur = audio.size / RATE
    if 0 <= held_ms < MIN_CLIP_S * 1000:
        return "", f"too_short held={held_ms:.0f}ms", 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < (VAD_MIN_CLIP_RMS if vad_speech_s is not None else MIN_RMS):
        return "", f"too_quiet rms={rms:.4f}", 0.0
    t0 = now_ms()
    pairs = parse_hints(hints)
    prompt = hint_prompt(pairs)
    # Older/offline transcribers can opt out; replacement still works without prompt support.
    prompted = getattr(tr, "supports_initial_prompt", False)
    result = (tr.run(audio, initial_prompt=prompt) if prompted
              else tr.run(audio))
    require_clear_speech = vad_speech_s is not None and (vad_speech_s < MIN_CLIP_S or rms < VAD_WEAK_CLIP_RMS)
    text, reason = checked_transcript(result, require_clear_speech)
    if text:
        text = collapse_transcript_runs(replace_hints(text, pairs))
        if is_unclear_transcript(text, pairs):
            text, reason = "", "unclear_short"
            if prompted:
                retry = tr.run(audio)
                candidate, _ = checked_transcript(retry, require_clear_speech=True)
                candidate = collapse_transcript_runs(replace_hints(candidate, pairs))
                if candidate and not is_unclear_transcript(candidate, pairs):
                    result, text, reason = retry, candidate, ""
    if raw_sink is not None and reason != "no_speech":
        raw_sink.append(result.get("text", ""))
    return text, reason or f"ok rms={rms:.4f} dur={dur:.2f}s", now_ms() - t0


# ---------------------------------------------------------------- world link

def elapsed_ms(start, end):
    return round((end - start) * 1000) if start is not None and end is not None else -1


class TimedReply(dict):
    def __init__(self, payload, job, final=True):
        super().__init__(payload)
        chat = getattr(job, "chat", None)
        if chat:            # every body of a text-chat turn says which request and whose bubble it answers
            self.update(req=chat["req"], by=chat["by"])
        self.vad_end_at = getattr(job, "vad_end_at", None)
        self.tail, self.npc_epoch, self.final = job.tail, getattr(job, "npc_epoch", None), final


class SearchReply(TimedReply):
    def __init__(self, payload, job, final=False, speaker=""):
        super().__init__(payload, job, final=final)
        self.speaker = speaker

    def valid(self):
        npc = self["npc"]
        return self.tail.npc_epoch[npc] == self.npc_epoch and self.tail.npc_state[npc]


class WaitReply(TimedReply):
    """The "잠깐만, 생각 중이야" line of a slow answer (TODO 135/140). If the answer is queued before the world
    took this line, the line is dropped: it must never make the answer wait for the next 5 s poll."""

    def __init__(self, payload, job):
        super().__init__(payload, job, final=False)


class DialogReply(TimedReply):
    def __init__(self, row, job, batch, index, previous, more):
        super().__init__(dict(row, seq=job.seq, more=more), job)
        self.tail, self.epochs, self.batch, self.previous = job.tail, dict(job.tail.npc_epoch), batch, previous
        self.index = index
        gesture = self.pop("gesture", "none")
        self["steps"] = [{"do": "gesture", "gesture": gesture}]

    def valid(self):
        return self.tail.npc_epoch == self.epochs and all(self.tail.npc_state.values())


class ReplyQueue(queue.Queue):
    # W04Probe keeps one pending body per NPC for 11 s. A later body could replace a search result.
    search_play_window = 11.5

    def __init__(self):
        super().__init__()
        self.protected_until = {}
        self.statuses = {}
        self.last_served = None
        self.ai_times = collections.deque(maxlen=20)
        self.typical_ai_s = 1.8

    def record_ai_time(self, ai_ms):
        with self.not_empty:
            self.ai_times.append(ai_ms / 1000)
            self.typical_ai_s = min(4.0, max(1.0, statistics.median(self.ai_times)))
            self.not_empty.notify_all()

    def poll_served(self):
        with self.not_empty:
            self.last_served = time.monotonic()

    def status(self, state, tail, npc, epoch, speaker):
        if tail is None:
            return
        with self.not_empty:
            key = tail, npc, state
            self.statuses.pop(key, None)
            self.statuses[key] = epoch, time.monotonic(), {"state": state, "speaker": speaker}
            self.not_empty.notify_all()

    def clear_status(self, tail, npc, state=None, epoch=None):
        if tail is None:
            return
        with self.mutex:
            self._clear_status(tail, npc, state, epoch)

    def _clear_status(self, tail, npc, state=None, epoch=None):
        for key, (generation, _, _) in tuple(self.statuses.items()):
            if (key[1] == npc and (tail is None or key[0] is tail)
                    and (state is None or key[2] == state) and (epoch is None or generation == epoch)):
                del self.statuses[key]

    def _put(self, reply):
        if isinstance(reply, TimedReply) and reply.final and reply.tail is not None:
            npc = reply["npc"]
            epoch = reply.epochs[npc] if isinstance(reply, DialogReply) else reply.npc_epoch
            self._clear_status(reply.tail, npc, epoch=epoch)
            if any(isinstance(item, WaitReply) for item in self.queue):
                self.queue = collections.deque(
                    item for item in self.queue
                    if not (isinstance(item, WaitReply) and item.get("npc") == npc and item.get("seq") == reply.get("seq")))
        super()._put(reply)

    def discard(self, keep):
        with self.mutex:
            self.queue = collections.deque(reply for reply in self.queue if keep(reply))

    def get_for_world(self, timeout, include_state=False):
        deadline = time.monotonic() + timeout
        with self.not_empty:
            while True:
                now = time.monotonic()
                wake = deadline
                for key, (epoch, started, _) in tuple(self.statuses.items()):
                    tail, npc, _ = key
                    if started + 10 <= now or epoch != tail.npc_epoch[npc] or not tail.npc_state[npc]:
                        del self.statuses[key]
                # An outdated search/dialog reply never becomes valid again (its epoch moved on): drop it
                # instead of skipping it on every poll.
                kept = [reply for reply in self.queue
                        if not (isinstance(reply, (SearchReply, DialogReply)) and not reply.valid())]
                if len(kept) != len(self.queue):
                    self.queue.clear()
                    self.queue.extend(kept)
                for i, reply in enumerate(self.queue):
                    npc = reply.get("npc")
                    ready = self.protected_until.get(npc, (None, 0))[1]
                    if isinstance(reply, DialogReply):
                        if reply.index != reply.batch["next"]:
                            continue
                        ready = max(ready, reply.batch["ready"])
                    if ready > now:
                        wake = min(wake, ready)
                        continue
                    del self.queue[i]
                    if isinstance(reply, SearchReply) and reply.final:
                        self.protected_until[npc] = reply["seq"], now + self.search_play_window
                    if isinstance(reply, DialogReply):
                        reply.batch["next"] += 1
                        reply.batch["ready"] = now + min(12, max(5.5, 1 + len(reply["line"]) * 0.1))
                    self._clear_status(getattr(reply, "tail", None), npc)
                    return reply
                if include_state:
                    # Estimate the next request conservatively from the previous send.
                    next_poll_at = now if self.last_served is None else self.last_served + POLL_MIN_INTERVAL_S
                    for key, (_, started, status) in tuple(self.statuses.items()):
                        overdue = started + STATUS_WORTH_S
                        remaining = started + self.typical_ai_s - max(now, next_poll_at)
                        if key[2] == "thinking" and now < overdue and remaining <= STATUS_WORTH_S:
                            wake = min(wake, overdue)
                            continue
                        del self.statuses[key]
                        return status
                if now >= deadline:
                    raise queue.Empty
                self.not_empty.wait(max(0, wake - now))

    def retry_search(self, reply):
        if not isinstance(reply, SearchReply):
            return
        npc = reply["npc"]
        with self.not_empty:
            if not reply.valid():
                return
            if reply.final and self.protected_until.get(npc, (None, 0))[0] == reply["seq"]:
                del self.protected_until[npc]
            self.queue.appendleft(reply)
            self.not_empty.notify_all()


replies = ReplyQueue()
choice_mailbox = ChoiceMailbox()
_seq = 0
UNBOUND = object()      # answer(): "no dictation decision was attached to this job" (push-to-talk)


def clear_replies(preserve_search=False):
    replies.discard(lambda reply: preserve_search and isinstance(reply, SearchReply))


@dataclass
class Utterance:
    seq: int
    t_up: float
    held_ms: float
    audio: np.ndarray
    npc: str
    by_voice: bool
    tail: object
    npc_epoch: int
    speaker: str = ""
    dictate: object = UNBOUND
    started: float = 0.0
    ended: float = 0.0
    vad_speech_s: float | None = None
    choice: dict | None = None
    chat: dict | None = None            # a visitor's text chat (CHAT_REQ): {"req", "by", "text"}
    mute_epoch: int = 0
    vad_end_at: float | None = None
    queued_at: float | None = None
    stt_start_at: float | None = None
    stt_end_at: float | None = None
    merge_wait_ms: int = 0
    ai_start_at: float | None = None
    ai_end_at: float | None = None
    reply_queued_at: float | None = None
    brain_init_ms: int = 0
    timing_logged: bool = False

    def context(self):
        return self.tail, self.npc, self.npc_epoch, self.speaker

    def log_timing(self):
        stages = (("queue", self.queued_at), ("stt_start", self.stt_start_at), ("stt_end", self.stt_end_at),
                  ("ai_start", self.ai_start_at), ("ai_end", self.ai_end_at), ("reply_queued", self.reply_queued_at))
        offsets = "\t".join(f"{stage}_ms={elapsed_ms(self.vad_end_at, at)}" for stage, at in stages)
        log("TIMING", f"seq={self.seq}\tvad_end_ms={0 if self.vad_end_at is not None else -1}\t{offsets}\t"
                      f"merge_wait_ms={self.merge_wait_ms}\tbrain_init_ms={self.brain_init_ms}")
        self.timing_logged = True


def step_note_text(key: str) -> str:
    """The Korean note the player heard for a step the code changed or left out (activity "drop" row)."""
    text = persona_ai.STEP_NOTES.get(key, ("",))[0]
    return "" if "{" in text else text


def make_empty_reply(target: str, seq: int) -> dict:
    """Answer for a push-to-talk that was dictated into the settings panel: no "line", so the world
    treats it like the 55 s timeout answer {} (POLL_EMPTY) and just closes the thinking dots."""
    return {"line": "", "dictate": target, "seq": seq}


def make_reply(text: str, stt_ms: float) -> dict:
    """The echo reply (no AI). The line is cut to the world's limit; "heard" keeps the whole transcript."""
    line = HEARD_FMT.format(text=text) if text else MISSED_LINE
    if len(line) > WORLD_MAX_LINE:
        line = line[:WORLD_MAX_LINE - 1] + "…"
    return {
        "emotion": "neutral" if text else "troubled",
        "intensity": 0.5,
        "steps": [{"do": "look_at", "target": "speaker"}],
        "line": line,
        "heard": text,
        "stt_ms": round(stt_ms),
        "seq": _seq,
    }


def poll_output(reply):
    """The body the world gets for one queued reply. Its activity record ("act") rides along only while
    the whole body stays within activity.MAX_BODY_BYTES (oldest rows left out first, then the act)."""
    act = reply.get("act")
    if act is None:
        return _poll_body(reply)
    body = {key: value for key, value in _poll_body(reply).items() if key != "act"}
    return activity.fit_body(body, act, dumps=lambda value: json.dumps(value, ensure_ascii=False))


def _poll_body(reply):
    if isinstance(reply, SearchReply) and not reply.final:
        return {"state": "searching", "speaker": reply.speaker,
                **{key: reply[key] for key in ("req", "by") if key in reply}}
    line = reply.get("line")
    if not line or reply.get("lines"):
        return reply
    if line == MISSED_LINE:
        text = (line, "Huh? I didn't catch that.", "ん？よく聞こえなかったよ。")
    elif line.startswith("들은 말: "):
        heard = reply.get("heard", "")
        language_reply = persona_ai.output_language_reply(heard)
        if language_reply is not None:
            return dict(reply, **language_reply)
        text = (line, "I heard: " + heard, "聞こえた言葉： " + heard)
    else:
        return reply
    return dict(reply, lines=persona_ai.localized_lines(
        tuple(s if len(s) <= WORLD_MAX_LINE else s[:WORLD_MAX_LINE - 1] + "…" for s in text)))


class Handler(BaseHTTPRequestHandler):
    """/poll and /ping are for the world (GET only, no token: unchanged).
    /cfg is for the world's settings panel: no token, open only shortly after the panel says CFG_OPEN.
    /settings and /api/* are for the user's browser and need the per-run token."""

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: str, code: int = 200, ctype: str = "text/plain") -> bool:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if ctype != "text/plain" or code != 200:
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
        if ctype == "text/html":
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                             "connect-src 'self'; base-uri 'none'; form-action 'none'")
        try:
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True
            return False
        return True

    # ---- settings: who may come in

    def _local_names(self):
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _allowed(self, query: str = "") -> bool:
        """Token (header, or ?t= for the page itself) + Host/Origin must be this PC's own address."""
        token = getattr(self.server, "token", "")
        if not token:
            return False
        if (self.headers.get("Host") or "").lower() not in self._local_names():
            return False                       # blocks DNS rebinding: another site's name pointed at 127.0.0.1
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in {"http://" + h for h in self._local_names()}:
            return False
        given = self.headers.get("X-W04-Token") or ""
        if not given and query:
            m = re.search(r"(?:^|&)t=([A-Za-z0-9_\-]+)", query)
            given = m.group(1) if m else ""
        return hmac.compare_digest(given.encode(), token.encode())

    def _deny(self):
        self._send("이 주소는 앱 창에 나온 '설정 화면' 주소(토큰 포함)로만 열 수 있습니다. "
                   "settings.cmd를 더블클릭하거나 앱 창의 주소를 복사해 주세요.", 403)

    def _state_json(self, npc=None) -> str:
        npc = persona_store.npc_id(npc)
        store = self.server.store.for_npc(npc)
        brain = select_brain(getattr(self.server, "brain", None), npc)
        state = store.state()
        state["settings"].update(store.brain_options())
        state["brain_access"] = self.server.brain_settings.brain_access(store.speaker if store.speaker_known else "")
        other = self.server.store.for_npc(next(key for key in persona_store.NPC_IDS if key != npc))
        with other.lock:
            state["other_impressions"] = {"npc": other.npc_name(), "players": [
                {"name": other.player_label(name), "note": entry["note"]}
                for name, entry in sorted(other.players.items()) if entry["note"]
            ]}
        state["search_links"] = search_links.recent.snapshot(store.root_dir, npc)
        state["ai"] = brain.model if brain is not None else ""
        state["rules"] = persona_ai.RULES
        state["system_preview"] = (brain.system_prompt() if brain is not None
                                   else persona_ai.RULES + "\n" + state["persona"].strip() + "\n\n"
                                   + store.prompt_tail(store.speaker))
        return json.dumps(state, ensure_ascii=False)

    def do_GET(self):
        t0 = now_ms()
        path, _, query = self.path.partition("?")
        log("HTTP_IN", path)                   # the query (token) is never written to the log
        choice_npc = path.removeprefix("/choice/") if path.startswith("/choice/") else ""
        if path in ("/poll", "/ping") or path.startswith("/ping/") or choice_npc in persona_store.NPC_IDS:
            # The world asks with this PC's own address and never sends Origin. Another site's name pointed
            # at 127.0.0.1 (DNS rebinding) or a web page's cross-site request is turned away BEFORE a reply
            # is taken from the queue, so it can neither read nor swallow an answer.
            if ((self.headers.get("Host") or "").lower() not in self._local_names()
                    or self.headers.get("Origin") is not None
                    or (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("none", "same-origin")):
                self._send("{}", 403, "application/json")
            elif choice_npc:
                self._send(json.dumps(choice_mailbox.take(choice_npc), ensure_ascii=False), ctype="application/json")
            elif path == "/poll":
                try:
                    reply = replies.get_for_world(POLL_WAIT_S, include_state=True)
                except queue.Empty:
                    if self._send("{}"):
                        replies.poll_served()
                else:
                    outgoing = poll_output(reply)
                    try:
                        sent = self._send(json.dumps(outgoing, ensure_ascii=False))
                    except OSError:
                        replies.retry_search(reply)
                    else:
                        if not sent:
                            replies.retry_search(reply)
                            return
                        replies.poll_served()
                        try:
                            if isinstance(reply, DialogReply):
                                reply.batch["chat"].remember(reply, reply.previous)
                            for row in outgoing.get("dialog") or [outgoing]:
                                if row.get("line"):
                                    lines = row.get("lines") or {}
                                    since_vad = elapsed_ms(getattr(reply, "vad_end_at", None), time.monotonic())
                                    log("POLL_LINES", f"npc={row.get('npc', '')}\tseq={outgoing.get('seq', '')}\t"
                                                      f"since_vad_ms={since_vad}\t"
                                                      f"en={int(bool(lines.get('en')))}\tja={int(bool(lines.get('ja')))}")
                        except OSError as e:
                            # log() never raises OSError itself: this is the small-talk memory write.
                            log("POLL_REMEMBER_ERROR", type(e).__name__, say="NPC 잡담 기록을 저장하지 못함")
            else:
                self._send(f"pong {t0:.0f}")
        elif path == "/cfg":
            bridge = getattr(self.server, "bridge", None)
            # Host must be this PC's own address and a browser page (it sends Origin) is turned away
            if (bridge is None or (self.headers.get("Host") or "").lower() not in self._local_names()
                    or self.headers.get("Origin") is not None or not bridge.wait_open()):
                self._send('{"ok":false}', 403, "application/json")
            else:
                # no ?npc= (an older panel) = kwonjun, never "whoever was opened last" (contract.md)
                npc = persona_store.npc_id(parse_qs(query).get("npc", [None])[0])
                self._send(bridge.payload(npc).decode("utf-8"), ctype="application/json")
        elif path == "/settings":
            if not self._allowed(query):
                self._deny()
            else:
                with open(SETTINGS_HTML, encoding="utf-8") as f:
                    self._send(f.read().replace("__TOKEN__", self.server.token), ctype="text/html")
        elif path == "/api/state":
            if not self._allowed():            # header only: a world can add a query but not a header
                self._deny()
            else:
                self._send(self._state_json(parse_qs(query).get("npc", [None])[0]), ctype="application/json")
        else:
            self._send("unknown")
        log("HTTP_OUT", f"{path}\theld_ms={now_ms() - t0:.0f}")

    def do_POST(self):
        path = self.path.partition("?")[0]
        log("HTTP_IN", "POST " + path)
        if path != "/api/save" or not self._allowed():
            self._deny()
            return
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_POST_BYTES:
                raise ValueError("너무 큰 요청입니다.")
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise ValueError("bad content type")
            req = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("bad request")
            npc = persona_store.npc_id(req.get("npc"))
            st = self.server.store.for_npc(npc)
            settings = req.get("settings")
            model_patch = {}
            if isinstance(settings, dict):
                model_patch = {key: settings.pop(key) for key in ("brain_model", "brain_effort") if key in settings}
                if not settings:
                    del req["settings"]
            if model_patch or "brain_unlocked" in req:
                if "brain_unlocked" in req and type(req["brain_unlocked"]) is not bool:
                    raise ValueError("모델 잠금 값이 올바르지 않습니다.")
                self.server.brain_settings.change_brain(st.speaker if st.speaker_known else "",
                                                        model_patch, req.get("brain_unlocked"))
            st.apply(req)
            if req.get("compact") == "undo":
                err = st.compaction_undo()
            elif req.get("compact") == "run":
                err = start_compaction(st, select_brain(getattr(self.server, "brain", None), npc), force=True)
            else:
                err = ""
            if err:
                raise ValueError(err)
        except (ValueError, OSError) as e:
            self._send(json.dumps({"ok": False, "error": str(e)[:200]}, ensure_ascii=False), 400, "application/json")
            return
        log("SETTINGS", "saved", say="설정 화면에서 저장함 (다음 대답부터 적용)")
        self._send(self._state_json(npc), ctype="application/json")


class ExclusiveServer(ThreadingHTTPServer):
    """Refuses a port another app already holds. Python's default (SO_REUSEADDR) lets a second copy
    of this app share the port on Windows, so two apps heard the same voice and both answered."""
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):          # Windows only
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(port: int, store, brain=None, token: str = None, bridge=None) -> ThreadingHTTPServer:
    """The one HTTP server (127.0.0.1 only): /poll and /cfg for the world, /settings for the user.
    Raises OSError when the port is taken (usually: this app is already running)."""
    server = ExclusiveServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.store, server.brain, server.bridge = store, brain, bridge
    server.brain_settings = bridge or (cfg_bridge.CfgBridge(store) if store is not None else None)
    server.token = token or secrets.token_urlsafe(24)
    server.settings_url = f"http://127.0.0.1:{server.server_address[1]}/settings?t={server.token}"
    return server


def latest_output_log(log_dir: str):
    files = glob.glob(os.path.join(log_dir, "output_log_*.txt"))
    return max(files, key=os.path.getmtime) if files else None


def recent_output_logs(log_dir: str, within_s: float = WATCH_RECENT_S) -> list:
    """Every output_log written lately (one per running VRChat client), plus the newest one."""
    out, newest, newest_t = [], None, -1.0
    for path in glob.glob(os.path.join(log_dir, "output_log_*.txt")):
        try:
            t = os.path.getmtime(path)
        except OSError:
            continue
        if t > newest_t:
            newest, newest_t = path, t
        if time.time() - t <= within_s:
            out.append(path)
    if newest and newest not in out:
        out.append(newest)
    return sorted(out)


class LogTail:
    """One output_log file: its own read position and the world its client is in right now.

    The existing file is read once for the world, the player name and read-only world state.
    Old [W04AI] commands in that part are never acted on."""

    def __init__(self, path: str):
        self.path = path
        self.world = None
        self.player = ""
        self.npc_state = dict(DEFAULT_NPC_STATE)
        self.npc_epoch = dict.fromkeys(DEFAULT_NPC_STATE, 0)
        self.last_answer = {}
        self.chat_epoch = 0
        self.global_mode = False
        self.content_level = 0
        self.playback = tv_history.PlaybackLog()
        self.reset_world_state()
        self.command_results = {}
        self.clarifications = {}
        self.buf = b""
        self.pos = 0
        self.fh = open(path, "rb")
        end = os.path.getsize(path)
        while self.fh.tell() < end:
            try:
                raw = self.fh.readline()
            except OSError:
                return
            if not raw.endswith(b"\n"):                 # half-written last line: read it again later, live
                self.fh.seek(-len(raw), os.SEEK_CUR)
                break
            line = raw.decode("utf-8", errors="replace")
            world = cfg_bridge.world_from_line(line, self.world)
            if world != self.world:
                self.npc_state = dict(DEFAULT_NPC_STATE)
                self.playback = tv_history.PlaybackLog()
            if world != self.world or ("Joining " in line and cfg_bridge.JOIN_RE.match(line)):
                self.reset_world_state()
            self.world = world
            self.playback.feed(line)
            update_npc_state(self.npc_state, line)
            event = EVENT_RE.search(line)
            if event and event[1] in WORLD_STATE_EVENTS:
                update_world_state(self, event[1], line)
            if "Initialized PlayerAPI" in line:
                m = LOCAL_PLAYER_RE.search(line)
                if m:
                    self.player = m.group(1)

    def reset_world_state(self):
        self.tv_state = {}
        self.mirror_state = {}
        self.activity_state = {}

    def park(self):
        """Not written for a while: let go of the file (VRChat deletes old logs) but keep the position."""
        if self.fh is not None:
            self.pos = self.fh.tell()
            self.close()

    def resume(self):
        if self.fh is None:
            self.fh = open(self.path, "rb")
            self.fh.seek(self.pos)

    def lines(self):
        """New complete lines since the last call."""
        while self.fh is not None:
            raw = self.fh.readline()
            if not raw:
                return
            self.buf += raw
            if not self.buf.endswith(b"\n"):
                return
            line, self.buf = self.buf.decode("utf-8", errors="replace"), b""
            yield line

    def close(self):
        try:
            if self.fh is not None:
                self.fh.close()
        except OSError:
            pass
        self.fh = None


def find_local_player(path: str) -> str:
    """Display name of the VRChat user on this PC, from lines already in the log ('' if none yet)."""
    try:
        tail = LogTail(path)
    except OSError:
        return ""
    tail.close()
    return tail.player


def listen_mode_from_env() -> str:
    mode = os.environ.get("W04_LISTEN_MODE", "").strip().lower()
    return mode if mode in LISTEN_MODES else "gaze"


def watch_log(log_dir: str, rec, tr: Transcriber, save: bool, brain=None, store=None, bridge=None, stop=None,
              listen_mode: str = None, mute=None):
    """Follows every recent output_log at once. [W04AI] lines count only from a log whose client is in
    an allowed world (bridge.world_allowed). stop: a threading.Event that ends the loop (tests).
    listen_mode: "gaze" (LISTEN_ON/OFF open the mic to the voice detector; jump still works) or "ptt"."""
    global _seq
    store = store or (brain.store if brain is not None else persona_store.Store())
    bridge = bridge or cfg_bridge.CfgBridge(store, log=log)
    if isinstance(brain, NpcBrains):
        def summarize_video(*args):
            summary_brain = brain.for_npc("kwonjun")
            if getattr(summary_brain, "client", None):
                return summary_brain.summarize_video(*args)

        summarize = summarize_video if brain.mode in MODELS and os.environ.get("ANTHROPIC_API_KEY") else None
    else:
        summarize = getattr(brain, "summarize_video", None) if getattr(brain, "client", None) else None
    cache = tv_history.VideoCache(store.root_dir, os.path.join(log_dir, "Tools", "yt-dlp.exe"),
                                 summarize)
    listen_mode = listen_mode or listen_mode_from_env()
    if listen_mode == "gaze" and not hasattr(rec, "open_tap"):
        listen_mode = "ptt"                     # this microphone stand-in cannot stream
    gaze = listen_mode == "gaze"
    # gaze state, shared by the watcher (log lines), the listener (audio) and the worker (answers)
    state_lock = threading.Lock()
    # which log switched listening on, its last line, and who that client's player is
    listen = {"on": False, "path": None, "seen": 0.0, "npc": "kwonjun", "speaker": ""}
    turn = {"pending": 0, "ptt": False, "gaze_active": None, "speech_epoch": 0, "npc": "kwonjun",
            "stage": "idle"}
    mute = mute or osc_mute.MuteState()
    chat_inflight = False
    chats = None

    def mic_valid(epoch):
        muted, current = mute.snapshot
        return not muted and epoch == current

    def explicit_chat(job, previous=None):
        if job.tail is None or not all(job.tail.npc_state.values()):
            return False
        with state_lock:
            epochs = dict(job.tail.npc_epoch)
            speech = turn["speech_epoch"]

        def valid():
            return ((stop is None or not stop.is_set()) and job.tail.npc_epoch == epochs
                    and epochs[job.npc] == job.npc_epoch and all(job.tail.npc_state.values())
                    and turn["speech_epoch"] == speech)

        chat = smalltalk.SmallTalk(store)
        result, source = chat.reply(select_brain(brain, job.npc), valid,
                                    job.tail.global_mode, job.tail.content_level,
                                    requested_by_player=True, opener=previous)
        if result is None:
            return True
        rows = result["dialog"]
        batch = {"ready": time.monotonic() + (5.5 if previous else 0), "chat": chat, "next": 0}
        with state_lock:
            if not valid():
                return True
            job.reply_queued_at = time.monotonic()
            for index, row in enumerate(rows):
                replies.put(DialogReply(row, job, batch, index, previous, index + 1 < len(rows)))
                previous = row
        log("NPC_CHAT", f"source={source}\trequested=1\tlines={len(rows)}")
        return True

    def player_busy():
        return turn["ptt"] or turn["gaze_active"] or turn["pending"] > 0

    def request_smalltalk(tail):
        nonlocal chat_inflight
        with state_lock:
            if chat_inflight or player_busy() or not all(tail.npc_state.values()) or not replies.empty():
                return
            chat_inflight = True
            speech_epoch, chat_epoch = turn["speech_epoch"], tail.chat_epoch

        def valid():
            return ((stop is None or not stop.is_set()) and not player_busy() and replies.empty()
                    and turn["speech_epoch"] == speech_epoch and tail.chat_epoch == chat_epoch)

        def still_idle():
            with state_lock:
                return valid()

        def generate():
            nonlocal chat_inflight, chats
            reply = None
            try:
                if not still_idle():
                    return
                if chats is None:
                    chats = smalltalk.SmallTalk(store)
                try:
                    selected = select_brain(brain, "kwonjun")
                except OSError:
                    selected = None
                reply, source = chats.reply(selected, still_idle, global_mode=tail.global_mode,
                                            content_level=tail.content_level)
            except (OSError, ValueError, KeyError, TypeError) as error:
                log("NPC_CHAT", f"unavailable={type(error).__name__}")
            finally:
                with state_lock:
                    chat_inflight = False
                    published = reply is not None and valid()
                    if published:
                        replies.put(reply)
            if published:
                log("NPC_CHAT", f"source={source}\tlines={len(reply['dialog'])}")

        threading.Thread(target=generate, daemon=True).start()

    def listen_off(reason: str, say: str = "듣기 멈춤"):
        with state_lock:
            was, listen["on"], listen["path"] = listen["on"], False, None
        if was:
            log("LISTEN_OFF", reason, say=say)

    def speaker_of(tail) -> str:
        """Whose client wrote this log (its own "Initialized PlayerAPI" line), else the last known name."""
        return (tail.player if tail is not None else "") or store.speaker

    def speaker_is(name: str):
        if name and (not store.speaker_known or name != store.speaker):
            store.set_speaker(name)
            log("SPEAKER", name, say=f"말하는 사람(내 VRChat 이름): {name}")

    work: "queue.Queue[Utterance]" = queue.Queue()

    def enqueue(job):
        job.queued_at = time.monotonic()
        work.put(job)

    def set_stage(stage):
        with state_lock:
            turn["stage"] = stage

    def done():
        with state_lock:
            turn["pending"] -= 1

    def failed(job, error):
        log("WORKER_ERROR", f"seq={job.seq}\t{exception_location(error)}\t{activity.log_code('V109')}",
            say=f"이번 말은 처리하지 못함 ({type(error).__name__}, V109). 다음 말부터 다시 받습니다.")
        replies.clear_status(job.tail, job.npc, "thinking", job.npc_epoch)
        trail = activity.Turn(job.npc, job.seq, job.vad_end_at)
        trail.add_code("V109", text=f"{activity.code_text('V109')} · {type(error).__name__}")
        if job.choice:
            choice_mailbox.put(job.npc, {"choice_id": job.choice["id"]})
        elif not job.by_voice:
            replies.put(TimedReply(dict(make_reply("", 0.0), npc=job.npc, seq=job.seq, act=trail.act()), job))

    def complete(job, prepared):
        set_stage("answer")
        try:
            answer(job, *prepared)
        except Exception as error:
            failed(job, error)
        finally:
            done()
            if not job.timing_logged:
                try:
                    job.log_timing()
                except OSError:
                    pass

    fragment = None

    def worker():
        nonlocal fragment
        while stop is None or not stop.is_set():
            set_stage("fragment" if fragment else "idle")
            try:
                job = work.get(timeout=0.05 if fragment else 0.1)
            except queue.Empty:
                if fragment:
                    previous, prepared = fragment
                    deadline = previous.ended + FRAGMENT_GAP_S
                    with state_lock:
                        active = turn["gaze_active"]
                        waiting = (active is not None and active[0] == previous.context()
                                   and active[1] <= deadline and active[2] is None)
                        expired = time.monotonic() >= deadline and not waiting and work.empty()
                    if expired:
                        fragment = None
                        previous.merge_wait_ms = elapsed_ms(previous.stt_end_at, time.monotonic())
                        complete(previous, prepared)
                continue

            if fragment:
                previous, prior = fragment
                if (not job.by_voice or job.dictate is not None or job.context() != previous.context()
                        or job.started > previous.ended + FRAGMENT_GAP_S):
                    fragment = None
                    previous.merge_wait_ms = elapsed_ms(previous.stt_end_at, time.monotonic())
                    complete(previous, prior)
            set_stage("stt" if job.by_voice and job.dictate is None else "answer")
            try:
                prepared = transcribe(job)
            except Exception as error:
                failed(job, error)
                done()
                continue
            if prepared is None:
                done()
                continue
            text, reason, stt_ms, clip, dictate, raw = prepared
            if fragment and text:
                previous, prior = fragment
                job.merge_wait_ms = elapsed_ms(previous.stt_end_at, job.stt_start_at)
                text = collapse_transcript_runs(f"{prior[0]} {text}")
                prepared = (text, reason, prior[2] + stt_ms, " ".join(filter(None, (prior[3], clip))),
                            dictate, prior[5] + "\n" + raw)
                done()
                fragment = None
                complete(job, prepared)
            elif job.by_voice and dictate is None and text and is_fragment(text):
                fragment = job, prepared
            else:
                complete(job, prepared)

    def transcribe(job):
        if job.choice or job.chat:
            text = (job.choice or job.chat)["text"]
            return text, "choice" if job.choice else "chat", 0.0, "", None, text
        if not mic_valid(job.mute_epoch):
            return
        dictate = job.dictate
        if dictate is UNBOUND:              # push-to-talk keeps its old rule: what is armed when it is handled
            dictate = bridge.dictate_target()
        elif dictate is not None and bridge.dictate_target() != dictate:
            # it was recorded FOR the panel and that dictation was cancelled, replaced or timed out while the
            # job waited: thrown away. It must never fall through to the AI as ordinary talk.
            log("STT", f"seq={job.seq}\tdictate_gone\tafter_release_ms={now_ms() - job.t_up:.0f}\t{activity.log_code('V108')}",
                say="받아쓰기가 취소됐거나 시간이 지나서 이번 말은 버림")
            return
        hint_store = store.for_npc(dictate.get("npc") if dictate else job.npc)
        with hint_store.lock:
            hints = hint_store.settings["stt_hints"]
        raw = []
        job.stt_start_at = time.monotonic()
        text, reason, stt_ms = judge(job.audio, job.held_ms, tr, hints, raw_sink=raw, vad_speech_s=job.vad_speech_s)
        job.stt_end_at = time.monotonic()
        if not mic_valid(job.mute_epoch):
            return
        clip = save_clip(job.audio, job.seq) if save else ""
        return text, reason, stt_ms, clip, dictate, raw[0] if raw else ""

    def answer(job, text, reason, stt_ms, clip, dictate, raw):
        typed = bool(job.choice or job.chat)     # no microphone: a world choice button or a visitor's chat
        if not typed and not mic_valid(job.mute_epoch):
            return
        seq, t_up, npc, by_voice = job.seq, job.t_up, job.npc, job.by_voice
        st = store.for_npc(npc)
        speaker = job.speaker or st.speaker
        if dictate:
            # the settings panel asked for this utterance: it goes to the persona / memory, not to
            # the AI. The transcript itself is not written to the log.
            log("STT", f"seq={seq}\t{reason}\tstt_ms={stt_ms:.0f}\tafter_release_ms={now_ms() - t_up:.0f}\t"
                       f"(dictate {dictate['target']} len={len(text)})\t{clip}",
                say=(f"들음 ({stt_ms:.0f} ms): 설정 창에 받아씀" if text
                     else f"못 알아들음 ({reason}) - 받아쓰기는 계속 기다림"))
            if text:
                bridge.dictate_apply(text, expected=dictate)
            if not by_voice:                # the world's push-to-talk poll is waiting for an answer
                job.reply_queued_at = time.monotonic()
                replies.put(TimedReply(dict(make_empty_reply(dictate["target"], seq), npc=npc), job))
                log("REPLY", f"seq={seq}\tafter_release_ms={now_ms() - t_up:.0f}\tdictate")
            return
        stt_failed = "" if text or job.choice else activity.stt_code(reason)
        # the code rides inside the reason field (log_report reads the STT fields by position)
        log("STT", f"seq={seq}\t{reason}{' ' + activity.log_code(stt_failed) if stt_failed else ''}\tstt_ms={stt_ms:.0f}\t"
                   f"after_release_ms={now_ms() - t_up:.0f}\t{speaker}\t{text}\t{clip}",
            say=(f"{speaker} 월드 선택지: {text}" if job.choice else f"{speaker} 월드 글 채팅: {text}" if job.chat else
                 f"{speaker} 마이크 ({stt_ms:.0f} ms): {text}" if text else f"못 알아들음 ({reason})"))
        # The activity record of this turn (memory only; the panel's "활동 기록" tab). A dictation above
        # never gets one: its transcript is for the panel, not for a record.
        trail = activity.Turn(npc, seq, job.vad_end_at or job.stt_start_at)
        if text:
            trail.add("heard", text, job.stt_end_at, None if typed else round(stt_ms))
        elif stt_failed and not by_voice:
            # push-to-talk only: an open microphone hears noise all the time, which is not worth a row
            trail.add_code(stt_failed, at=job.stt_end_at)
        selected_brain = select_brain(brain, npc) if text else None
        lazy_brains = brain if isinstance(brain, NpcBrains) else getattr(brain, "npc_brains", None)
        if text and not job.choice and isinstance(lazy_brains, NpcBrains):
            job.brain_init_ms = lazy_brains.take_init_ms(npc)
        searching = False
        other = next(key for key in persona_store.NPC_IDS if key != npc)
        last_epoch, last_speaker, last_at = job.tail.last_answer.get(npc, (-1, "", 0)) if job.tail else (-1, "", 0)
        follow_up = (last_epoch == job.npc_epoch and last_speaker == speaker and 0 <= t_up - last_at <= FOLLOW_UP_MS)
        named = bool(text and npc_named(text, npc, st.npc_name()))

        def search_valid():
            return (job.tail.npc_epoch[npc] == job.npc_epoch and job.tail.npc_state[npc]
                    and (stop is None or not stop.is_set()))

        def log_ai_timing(ai):
            job.log_timing()
            fields = ("kind", "ms", "cache_read", "cache_write", "fp_system", "fp_schema", "thinking", "effort")
            calls = [":".join(str(call[field]) for field in fields)
                     + (f":retry={call['retry']}" if call.get("retry") else "") for call in ai.get("calls", ())]
            log("AI_CALLS", " ".join((f"seq={seq}", *calls)))

        def skip(ai, code="V110"):
            replies.clear_status(job.tail, npc, "thinking", job.npc_epoch)
            log("AI_SKIP", f"seq={seq}\tnpc={npc}\tai_ms={ai.get('ai_ms', 0)}\t"
                           f"named={int(named)}\trecent={int(follow_up)}\t넘김\t{activity.log_code(code)}",
                say=f"{st.npc_name()}: " + ("다른 NPC를 불러서 넘김" if code == "V111" else "다른 사람에게 한 말로 판단해 넘김"))
            trail.add_code(code, kind="skip")
            if not by_voice and search_valid():
                job.reply_queued_at = time.monotonic()
                replies.put(TimedReply({"npc": npc, "seq": seq, "line": "", "steps": [], "act": trail.act()}, job))
            log_ai_timing(ai)

        # Calling the other NPC by name is for that NPC. Only mentioning it ("이토아는 뭐 좋아해?") is still
        # for this one: the model's for_me check decides as usual (t128 A11).
        if text and not job.choice and not named and npc_called(text, other, store.for_npc(other).npc_name()):
            try:
                st.record_overheard()
            except OSError:
                pass
            skip({}, "V111")
            return

        chat_request = not job.choice and smalltalk.requested(text)
        if chat_request and explicit_chat(job):
            return
        command = local_commands.world_command(text)
        if (chat_request or command is not None) and search_valid():
            notice = (("둘이 이야기하려면 두 NPC를 켜 줘.", "Turn on both NPCs so we can talk.",
                       "二人で話せるように、両方のNPCをオンにしてね。") if chat_request
                      else ("알겠어.", "Got it.", "わかった。"))
            if command and command.pop("scope", None) == "all":
                notice = ("나이트 모드는 각자 화면에서만 바뀌어서 네 화면만 바꿨어.",
                          "Night mode changes only on each person's own screen, so I changed just yours.",
                          "ナイトモードは各自の画面だけで変わるから、あなたの画面だけ変えたよ。")
            lines = persona_ai.localized_lines(notice, job.tail.global_mode)
            reply = {"line": lines["ko"], "lines": lines, "steps": [command] if command else [],
                     "emotion": "neutral", "intensity": 0.5, "heard": text}
            selected_brain = None
        elif text and selected_brain is not None:
            if search_valid() and not job.choice:
                replies.status("thinking", job.tail, npc, job.npc_epoch, speaker)

            def interim(r):
                # web search turn: show "잠깐만, 찾아볼게!" now; "more" makes the world poll again for the answer
                nonlocal searching
                if r.get("interim") == "slow":
                    # The AI server did not answer the first try in time and it is being retried: put the
                    # "생각 중" status back (the world shows one for 10 s). ReplyQueue still holds it until
                    # 3 s have passed, so a quick retry answer is never kept waiting behind it.
                    if search_valid() and not job.choice:
                        replies.status("thinking", job.tail, npc, job.npc_epoch, speaker)
                    log("AI_RETRY", f"seq={seq}\t{activity.log_code('A102')}",
                        say=f"{st.npc_name()}: AI 서버가 느려 한 번 더 요청하는 중 (A102)")
                    trail.add_code("A102")
                    return
                if job.choice:
                    return
                if r.get("interim") == "wait":
                    # TODO 135/140: the answer is slow; one short "잠깐만, 생각 중이야" like the recall line. It is
                    # dropped from the queue if the answer arrives before the world took it (WaitReply).
                    if not search_valid():
                        return
                    first = {k: r[k] for k in ("emotion", "intensity", "steps", "lines", "line") if k in r}
                    first.update(heard=text, stt_ms=round(stt_ms), seq=seq, more=True, npc=npc)
                    trail.add_code("A101", kind="think")
                    first["act"] = trail.act()
                    replies.put(WaitReply(first, job))
                    log("REPLY", f"seq={seq}\tinterim\t{first['line']}\t{activity.log_code('A101')}",
                        say=f"{st.npc_name()}: {first['line']} (응답이 늦어 기다리는 중)")
                    return
                searching = r.get("interim") == "search"
                first = {k: r[k] for k in ("emotion", "intensity", "steps", "lines", "line") if k in r}
                first["steps"] = [step for step in first["steps"] if step["do"] in ("look_at", "gesture", "wait")]
                first.update(heard=text, stt_ms=round(stt_ms), seq=seq, more=True, npc=npc)
                if r.get("search_query"):
                    trail.add("search", r["search_query"])
                elif r.get("interim") == "recall":
                    trail.add("recall")
                first["act"] = trail.act()
                with state_lock:
                    if searching:
                        if not search_valid():
                            return
                        turn["stage"] = "search"
                    replies.put(SearchReply(first, job, speaker=speaker) if searching
                                else TimedReply(first, job, final=False))
                doing = "기억 되짚는 중" if r.get("interim") == "recall" else "검색 중"
                log("REPLY", f"seq={seq}\tinterim\t{first.get('line', '')}", say=f"{st.npc_name()}: {first.get('line', '')} ({doing})")

            options = {"global_mode": True} if job.tail.global_mode else {}
            if isinstance(selected_brain, persona_ai.Brain):
                def resolve_sizes(out):
                    with state_lock:
                        return resolve_tv_sizes(out, job.tail.tv_state, options.get("global_mode", False))

                captured_at = time.time() - max(0, now_ms() - t_up) / 1000
                with st.lock:
                    check_address = st.settings["filter_overheard"] and not (follow_up or named)
                with state_lock:
                    prior, until = job.tail.clarifications.pop(npc, ([], 0.0))
                    context = prior if time.monotonic() <= until else []
                    if job.choice:
                        previous, question = job.choice["context"]
                        context = [previous or (context[0] if context else ""), question]
                    tv_rows = cache.context(job.tail.playback, text, " ".join(context))
                    tv = tv_context(text, job.tail.tv_state)
                    mirrors = mirror_context(text, job.tail.mirror_state, npc)
                    activity_fact = activity_context(job.tail.activity_state, npc)
                    cmd_results = command_result_context(text, job.tail.command_results, npc)
                source = "월드 선택지" if job.choice else "월드 글 채팅" if job.chat else "PC 주인 마이크"
                options.update(content_level=job.tail.content_level, utterance={
                    "speaker": speaker, "source": source, "raw": raw,
                    "time": datetime.fromtimestamp(captured_at).astimezone().isoformat(timespec="seconds"),
                }, check_address=check_address and not context and not typed, resolve_tv_sizes=resolve_sizes)
                if context:
                    options["clarification"] = context
                situation = "\n".join(filter(None, (touch_context(npc), tv, mirrors, cmd_results, activity_fact)))
                if situation:
                    options["touch_now"] = situation
                if tv_rows:
                    options["tv_rows"] = tv_rows
            job.ai_start_at = time.monotonic()
            trail.add("think", at=job.ai_start_at)
            if options.get("tv_rows"):
                trail.add("tv", f"재생 기록 {len(options['tv_rows'])}개", job.ai_start_at)
            try:
                ai = selected_brain.reply(text, speaker=speaker, on_interim=interim, **options)
            finally:
                job.ai_end_at = time.monotonic()
            if not ai.get("error") and isinstance(ai.get("ai_ms"), (int, float)):
                replies.record_ai_time(ai["ai_ms"])
            if ai.get("for_me") is False:
                skip(ai)
                return
            trail.add_result(ai, job.ai_start_at, job.ai_end_at, step_text=step_note_text)
            reply = {k: ai[k] for k in ("emotion", "intensity", "steps", "lines", "line")}
            if "output_language" in ai:
                reply["output_language"] = ai["output_language"]
            if ai.get("choices"):
                reply["choices"] = ai["choices"]
            # The world uses the live pose when it starts a movement step, including getting up.
            reply.update(heard=search_links.hide_urls(text) if ai.get("search_query") else text,
                         stt_ms=round(stt_ms), ai_ms=ai["ai_ms"], seq=seq, npc=npc)
            err = ai.get("error", "")
            if ai.get("recall"):
                log("RECALL", f"seq={seq}\trecall_ms={ai.get('recall_ms', 0)}")
            lines = reply["lines"]
            # the error field starts with the code (A103 APITimeoutError: ...); empty when nothing failed
            err_field = f"{ai['code']} {err}" if err and ai.get("code") else err
            ai_detail = (f"seq={seq}\tmodel={ai['model']}\tai_ms={ai['ai_ms']}\t{ai.get('usage', '')}\t"
                      f"global={int(job.tail.global_mode)}\ten={translation_flag(lines, 'en')}\t"
                      f"ja={translation_flag(lines, 'ja')}\t{err_field}\t"
                      f"{reply['emotion']}\t{json.dumps(reply['steps'], ensure_ascii=False)}\t{reply['line']}")
            ai_say = (f"AI 오류 ({err_field})" + ai_error_hint(err) if err else
                      f"{st.npc_name()} ({ai['ai_ms'] / 1000:.1f}초, {reply['emotion']}): {reply['line']}")
        elif job.choice:
            reply = {}
        elif by_voice and not text:
            # open mic: a noise that was not speech must not make the NPC say "응? 잘 못 들었어."
            return
        else:
            reply = make_reply(text, stt_ms)
        reply.update(npc=npc, seq=seq)
        if not job.choice:
            trail.add("reply", reply.get("line", ""), ms=ai.get("ai_ms") if selected_brain is not None else None)
            reply["act"] = trail.act()
        reaction = (selected_brain is not None and not job.choice and not ai.get("error")
                    and all(job.tail.npc_state.values()) and smalltalk.addressed(reply["line"], store.for_npc(other)))
        if reaction:
            reply["more"] = True
        stale_search = False
        with state_lock:
            if job.choice:
                if search_valid():
                    job.reply_queued_at = time.monotonic()
                    choice_mailbox.put(npc, dict(reply, choice_id=job.choice["id"]))
            elif searching:
                stale_search = not search_valid()
                if not stale_search:
                    if turn["pending"] > 1 or turn["gaze_active"] or turn["ptt"]:
                        reply["more"] = True
                    job.reply_queued_at = time.monotonic()
                    replies.put(SearchReply(reply, job, final=True))
            else:
                job.reply_queued_at = time.monotonic()
                replies.put(TimedReply(reply, job))
            if reply.get("choices") and reply.get("line") and search_valid():
                job.tail.clarifications[npc] = ([text[:200], reply["line"]], time.monotonic() + 45)
            if selected_brain is not None and reply.get("line") and not ai.get("error") and search_valid():
                job.tail.last_answer[npc] = (job.npc_epoch, speaker, now_ms())
        if selected_brain is not None:
            log("AI", ai_detail, say=ai_say)
            log_ai_timing(ai)
        if stale_search:
            return
        log("REPLY", f"seq={seq}\tafter_release_ms={now_ms() - t_up:.0f}")
        if selected_brain is not None and ai.get("links"):
            search_links.recent.add(st.root_dir, npc, ai["links"])
            for link in ai["links"]:
                log("SEARCH_LINK", f"npc={npc}\t{link['title']}\t{link['url']}",
                    say=f"검색 출처 {link['title']} — {link['url']}")
        if reaction and search_valid():
            explicit_chat(job, {"npc": npc, "line": reply["line"]})
        if selected_brain is not None:      # the reply is already on its way: remember, then tidy if it grew
            try:
                if ai.get("error"):
                    start_compaction(st, selected_brain)
                else:
                    start_memory(st, selected_brain, speaker, force=persona_store.asks_to_remember(text))
            except Exception as e:
                log("MEMORY_CALL", f"npc={npc}\tok=0\t{type(e).__name__}")

    threading.Thread(target=supervise_thread, args=("처리", worker, stop), daemon=True).start()

    tap = None

    def listen_audio():
        """Gaze mode: while LISTEN is on (or a dictation is armed) the microphone goes through the
        voice detector. The tap closes as soon as listening ends and no utterance is in progress, so
        nothing is looked at outside those times."""
        global _seq
        nonlocal tap
        det = UtteranceDetector(getattr(rec, "rate", RATE))
        dropped = UtteranceDetector(det.rate)
        dropped_npc = None

        def record_dropped(npc, speech_s):
            if speech_s >= VAD_MIN_SPEECH_S:
                log("SPEECH_DROPPED", f"npc={npc}\treason=busy\tspeech_s={speech_s:.2f}\t"
                                      f"{activity.log_code(activity.note('V107'))}")

        utterance_tail = None
        utterance_npc, utterance_speaker, utterance_dictate = "kwonjun", "", None
        utterance_epoch, utterance_started = 0, 0.0
        audio_epoch = mute.snapshot[1]
        while stop is None or not stop.is_set():
            muted, current_epoch = mute.snapshot
            if muted or current_epoch != audio_epoch:
                audio_epoch = current_epoch
                if tap is not None:
                    rec.close_tap()
                    tap = None
                det.reset()
                if dropped.active:
                    record_dropped(dropped_npc, dropped.voiced * VAD_FRAME_S)
                dropped.reset()
                dropped_npc = None
                if utterance_tail is not None:
                    replies.clear_status(utterance_tail, utterance_npc, "hearing", utterance_epoch)
                with state_lock:
                    turn["gaze_active"] = None
                if muted:
                    time.sleep(0.02)
                    continue
            with state_lock:
                on, seen, current_npc, current_speaker = listen["on"], listen["seen"], listen["npc"], listen["speaker"]
                current_tail = listen.get("tail") if on else None
            if on and time.monotonic() - seen > LISTEN_STALE_S:
                listen_off("stale_log", say="듣기 멈춤 (VRChat 기록이 멈춤)")
                on = False
                current_tail = None
            armed = bridge.dictate_target()
            want = on or armed is not None
            if tap is None:
                if not want:
                    time.sleep(0.05)
                    continue
                tap = rec.open_tap()
                det.reset()
            try:
                block = tap.get(timeout=0.05)
            except queue.Empty:
                block = None
            if block is not None:
                blips = []
                dropped_speech = []
                with state_lock:
                    continuing = det.active
                    blocked = turn["pending"] and turn["stage"] == "answer"
                    dropping = bool(want and not turn["ptt"] and blocked and not continuing)
                    if dropped_npc is not None and (not dropping or dropped_npc != current_npc):
                        if dropped.active:
                            dropped_speech.append((dropped_npc, dropped.voiced * VAD_FRAME_S))
                        dropped.reset()
                        dropped_npc = None
                    if dropping:
                        # Measure dropped speech separately so it cannot enter the real VAD or STT.
                        dropped_npc = current_npc
                        dropped_speech.extend((dropped_npc, speech_s) for _, speech_s in dropped.feed(block))
                    if turn["ptt"] or (blocked and not continuing):
                        if continuing and utterance_tail is not None:
                            replies.clear_status(utterance_tail, utterance_npc, "hearing", utterance_epoch)
                        det.reset()  # speech during a player reply is dropped
                    else:
                        was_active = det.active
                        if not was_active:      # fixed when the speech starts, carried by the job
                            utterance_npc, utterance_speaker, utterance_dictate = current_npc, current_speaker, armed
                            utterance_tail = current_tail
                            utterance_epoch = current_tail.npc_epoch[current_npc] if current_tail else 0
                        finished = det.feed(block)
                        vad_end_at = time.monotonic() if finished else None
                        if not was_active and (det.active or finished):
                            turn["speech_epoch"] += 1
                            replies.discard(lambda reply: not isinstance(reply, DialogReply))
                            utterance_started = time.monotonic() - VAD_ONSET_VOICED * VAD_FRAME_S
                            if utterance_dictate is None:
                                replies.status("hearing", utterance_tail, utterance_npc,
                                               utterance_epoch, utterance_speaker)
                        for audio, speech_s in finished:
                            if utterance_tail is not None:
                                replies.clear_status(utterance_tail, utterance_npc, "hearing", utterance_epoch)
                            if speech_s < VAD_MIN_SPEECH_S:
                                blips.append((speech_s, det.floor, det.onset_threshold))
                                continue
                            if (utterance_dictate is None and
                                    (utterance_tail is None or not utterance_tail.npc_state[utterance_npc]
                                     or utterance_epoch != utterance_tail.npc_epoch[utterance_npc])):
                                continue
                            audio_16k = to_16k(audio, det.rate)
                            clear_replies(preserve_search=True)
                            _seq += 1
                            turn["pending"] += 1
                            ended = time.monotonic() - det.trailing_silence
                            log("VAD_END", f"seq={_seq}\tspeech_s={speech_s:.2f}\tclip_s={audio.size / det.rate:.2f}"
                                f"\tfloor_end={det.floor:.4f}\tonset_threshold_end={det.onset_threshold:.4f}",
                                say="말 끝남, 글자로 바꾸는 중...")
                            enqueue(Utterance(_seq, now_ms(), -1.0, audio_16k, utterance_npc, True,
                                              utterance_tail, utterance_epoch, utterance_speaker, utterance_dictate,
                                              utterance_started, ended, vad_speech_s=speech_s, mute_epoch=audio_epoch,
                                              vad_end_at=vad_end_at))
                            det.reset()
                            break
                    context = (utterance_tail, utterance_npc, utterance_epoch, utterance_speaker)
                    turn["gaze_active"] = (context, utterance_started, utterance_dictate) if det.active else None
                for speech_s, floor, threshold in blips:
                    log("VAD_BLIP", f"speech_s={speech_s:.2f}\tfloor_end={floor:.4f}\tonset_threshold_end={threshold:.4f}")
                for npc, speech_s in dropped_speech:
                    record_dropped(npc, speech_s)
            if not want and not det.active:
                if dropped.active:
                    record_dropped(dropped_npc, dropped.voiced * VAD_FRAME_S)
                dropped.reset()
                dropped_npc = None
                rec.close_tap()
                tap = None
                det.reset()

    def listener():
        nonlocal tap
        try:
            listen_audio()
        finally:
            with state_lock:
                active = turn["gaze_active"]
                turn["gaze_active"] = None
            if active and active[0][0] is not None:
                tail, npc, epoch, _ = active[0]
                replies.clear_status(tail, npc, "hearing", epoch)
            if tap is not None:
                tap = None
                rec.close_tap()

    if gaze:
        threading.Thread(target=supervise_thread, args=("듣기", listener, stop), daemon=True).start()

    def world_note(tail: LogTail):
        name = os.path.basename(tail.path)
        if tail.world is None:
            log("WORLD", f"{name}\tnone")
        elif bridge.world_allowed(tail.world):
            log("WORLD", f"{name}\t{tail.world}\tallowed", say="지금 월드: 허용된 월드 (점프로 말하기·월드 설정 창 사용 가능)")
        else:
            log("WORLD", f"{name}\t{tail.world}\tnot_allowed",
                say="지금 월드: 허용하지 않은 월드 (이 월드의 요청은 무시함. 내 월드라면 .env의 W04_WORLD_IDS에 ID를 적으세요)")

    def ptt_cancel(reason: str, path: str = None):
        """Drop a push-to-talk that is still held (path given: only if that log started it)."""
        with state_lock:
            if not turn["ptt"] or (path is not None and turn.get("path") != path):
                return
            turn["ptt"] = False
            replies.clear_status(turn["tail"], turn["npc"], "hearing", turn["npc_epoch"])
        if hasattr(rec, "cancel"):
            rec.cancel()
        else:
            rec.stop()
        code = "V113" if reason == "osc_muted" else "V115"
        log("PTT_CANCEL", f"{reason}\t{activity.log_code(code)}", say="녹음 취소 (월드를 떠났거나 점프 해제 기록이 없음)")

    def invalidate_search(tail, npc=None):
        # Called with state_lock, so a search cannot publish after its NPC was turned off.
        for key in (tail.npc_epoch if npc is None else (npc,)):
            tail.npc_epoch[key] += 1
            tail.clarifications.pop(key, None)
        replies.discard(lambda reply: not (isinstance(reply, SearchReply) and reply.tail is tail
                                          and (npc is None or reply["npc"] == npc))
                        and not (isinstance(reply, DialogReply) and reply.tail is tail))

    def handle(tail: LogTail, line: str):
        global _seq
        if bridge.world_allowed(tail.world) and ("[Video Playback]" in line or "[W04AI] TV_HISTORY" in line):
            with state_lock:
                cache.watch(tail.playback.feed(line))
            return
        if bridge.world_allowed(tail.world) and "[W04AI] CHOICE_" in line:
            event = read_event(line, "[W04AI] CHOICE_DONE ")
            if event:
                with state_lock:
                    tail.clarifications.pop(event["npc"], None)
                return
            event = read_event(line, "[W04AI] CHOICE_CONTEXT ")
            if event and tail.npc_state[event["npc"]]:
                with state_lock:
                    tail.clarifications[event["npc"]] = (event["context"], time.monotonic() + 45)
                return
            event = read_event(line, "[W04AI] CHOICE_INPUT ")
            if event and tail.npc_state[event["npc"]]:
                npc = event["npc"]
                with state_lock:
                    _seq += 1
                    turn["pending"] += 1
                    enqueue(Utterance(_seq, now_ms(), 0.0, None, npc, False, tail,
                                       tail.npc_epoch[npc], speaker_of(tail), choice=event))
            return
        if MARKER not in line:
            world = cfg_bridge.world_from_line(line, tail.world)
            if world != tail.world or ("Joining " in line and cfg_bridge.JOIN_RE.match(line)):
                tail.global_mode = False
                tail.content_level = 0
                with state_lock:
                    tail.playback = tv_history.PlaybackLog()
                    tail.reset_world_state()
                    tail.command_results.clear()
                    tail.clarifications.clear()
                    if world == tail.world:
                        invalidate_search(tail)
                clear_touch(tail.path)          # a hug in the old world never sends its AFFECTION_END
                if world == tail.world:
                    bridge.world_changed(tail.path)
            if world != tail.world:
                with state_lock:
                    tail.world = world
                    tail.npc_state = dict(DEFAULT_NPC_STATE)
                    tail.chat_epoch += 1
                    invalidate_search(tail)
                bridge.world_changed(tail.path)
                ptt_cancel("world_changed", tail.path)
                for npc_key in persona_store.NPC_IDS:
                    store.for_npc(npc_key).pose = "stand"
                if listen["path"] == tail.path:
                    listen_off("world_changed")
                world_note(tail)
            elif "Initialized PlayerAPI" in line:
                m = LOCAL_PLAYER_RE.search(line)
                if m:
                    tail.player = m.group(1)
                speaker_is(m.group(1) if m else "")
            return
        if bridge.handle_line(tail.path, tail.world, line, tail.player):
            return
        m = EVENT_RE.search(line)
        if not m:
            return
        event = m.group(1)
        npc = event_npc(line)
        if not bridge.world_allowed(tail.world):
            if event == "PTT_DOWN":
                log("PTT_IGNORED", f"world_not_allowed\t{activity.log_code(activity.note('V112'))}",
                    say="점프 표시를 무시함 (허용하지 않은 월드, V112)")
            elif event == "LISTEN_ON":
                log("LISTEN_IGNORED", f"world_not_allowed\t{activity.log_code(activity.note('V112'))}",
                    say="듣기 표시를 무시함 (허용하지 않은 월드, V112)")
            return
        if event in WORLD_STATE_EVENTS:
            with state_lock:
                update_world_state(tail, event, line)
            return
        if event == "CMD_RESULT":
            with state_lock:
                update_command_results(tail.command_results, line)
            return
        if event == "CONTENT_LEVEL":
            level = CONTENT_LEVEL_RE.match(line)
            if level:
                tail.content_level = int(level[1])
            return
        if event == "GLOBAL":
            flag = GLOBAL_RE.match(line)
            if flag:
                tail.global_mode = flag[1] == "1"
            return
        if event == "NPC_STATE":
            with state_lock:
                before = dict(tail.npc_state)
                update_npc_state(tail.npc_state, line)
                if before != tail.npc_state:
                    tail.chat_epoch += 1
                    for key, enabled in before.items():
                        if enabled and not tail.npc_state[key]:
                            invalidate_search(tail, key)
                off = listen["path"] == tail.path and not tail.npc_state[listen["npc"]]
            if off:
                listen_off("npc_disabled")
            return
        if event == "AFFECTION_END":
            m = AFFECTION_END_RE.search(line)
            if m:
                with touch_lock:
                    item = touch_now.get(npc)
                    if (item and item["type"] == "hug" and item["ended"] is None
                            and item["by"] == m.group(2)[:persona_store.MAX_NAME_CHARS]):
                        item["ended"] = time.monotonic()
            return
        if event == "AFFECTION":
            m = AFFECTION_RE.search(line)
            if m:
                kind, who = m.group(1), m.group(3)[:persona_store.MAX_NAME_CHARS]
                st = store.for_npc(npc)
                if kind in ("pat", "hug"):
                    with touch_lock:
                        now = time.monotonic()
                        item = touch_now.get(npc)
                        # A pat during an ongoing hug must not hide the hug or lose its end.
                        if (kind == "hug" or not item or item["type"] != "hug"
                                or item["ended"] is not None or now - item["at"] > TOUCH_HUG_MAX_S):
                            touch_now[npc] = {"type": kind, "by": who, "at": now, "path": tail.path}
                            if kind == "hug":
                                touch_now[npc]["ended"] = None
                phrase = st.add_affection(who, kind)
                log("AFFECTION", f"type={kind}\tnpc={npc}\tby={who}\tnoted={int(bool(phrase))}",
                    say=f"  └ {st.npc_name()}: {st.player_label(who)}에 대한 인상에 기록함: {phrase}" if phrase else "")
            return
        if event == "NPC_POSE":
            m = POSE_RE.search(line)
            if m:
                store.for_npc(npc).pose = m.group(2)
            return
        if event == "NPC_CHAT_REQ":
            if smalltalk.parse_request(line):
                request_smalltalk(tail)
            return
        if event == "CHAT_REQ":
            m = CHAT_REQ_RE.search(line)
            who = persona_store._one_line(m.group(3), persona_store.MAX_NAME_CHARS) if m else ""
            said = " ".join(m.group(4).split())[:CHAT_MAX_CHARS] if m else ""
            if not who or not said or not tail.npc_state[m.group(1)]:
                log("CHAT_REQ", "ignored", say="")
                return
            npc = m.group(1)
            with state_lock:
                _seq += 1
                turn["pending"] += 1
                job = Utterance(_seq, now_ms(), 0.0, None, npc, False, tail, tail.npc_epoch[npc], who,
                                chat={"req": int(m.group(2)) if m.group(2) else _seq, "by": who, "text": said})
            log("CHAT_REQ", f"seq={job.seq}\tnpc={npc}\treq={job.chat['req']}\tby={who}\tlen={len(said)}")
            enqueue(job)
            return
        if event in ("LISTEN_ON", "PTT_DOWN") and not tail.npc_state[npc]:
            return
        if event == "LISTEN_ON":
            if not gaze:
                log("LISTEN_IGNORED", "ptt_mode")
                return
            with state_lock:
                listen["on"], listen["path"], listen["seen"] = True, tail.path, time.monotonic()
                listen["tail"] = tail
                listen["npc"] = npc
                listen["speaker"] = speaker_of(tail)    # the player of THIS client, not "whoever joined last"
            log("LISTEN_ON", line[m.start():].strip(), say="NPC가 듣는 중 (그냥 말하면 됩니다)")
        elif event == "LISTEN_OFF":
            if listen["path"] == tail.path and listen["npc"] == npc:
                listen_off(line[m.start():].strip(), say="듣기 멈춤 (NPC를 안 보거나 멀어짐)")
        elif event == "PTT_DOWN":
            muted, epoch = mute.snapshot
            if muted:
                log("PTT_IGNORED", f"osc_muted\t{activity.log_code('V113')}")
                return
            with state_lock:
                clear_replies(preserve_search=True)
                turn["speech_epoch"] += 1
                turn["ptt"] = True
                turn["npc"] = npc
                turn["npc_epoch"] = tail.npc_epoch[npc]
                turn["mute_epoch"] = epoch
                turn["path"] = tail.path
                turn["tail"] = tail
                turn["down_at"] = time.monotonic()
                if bridge.dictate_target() is None:
                    replies.status("hearing", tail, npc, turn["npc_epoch"], speaker_of(tail))
            rec.start()
            log("PTT_DOWN", line.strip()[-80:], say="녹음 시작")
        else:
            # Ignore unmatched releases (including presses refused while this NPC was off).
            if not turn["ptt"] or turn.get("path") != tail.path:
                return
            if not mic_valid(turn["mute_epoch"]):
                ptt_cancel("osc_muted", tail.path)
                return
            m = re.search(r"held_ms=(\d+)", line)
            held_ms = float(m.group(1)) if m else -1.0
            log("PTT_UP", line.strip()[-80:], say="녹음 끝, 글자로 바꾸는 중...")
            with state_lock:
                target_npc = turn["npc"]
                npc_epoch = turn["npc_epoch"]
                replies.clear_status(tail, target_npc, "hearing", npc_epoch)
                _seq += 1
                seq = _seq
                turn["pending"] += 1
                turn["ptt"] = False
            # the recording is taken off the microphone right here (0.25 s post-roll), so a second press
            # while this one is still being answered can never swap its audio or its NPC
            job = Utterance(seq, now_ms(), held_ms, None, target_npc, False, tail, npc_epoch, speaker_of(tail),
                            mute_epoch=turn["mute_epoch"])
            try:
                job.audio = rec.stop()
            except Exception as e:
                with state_lock:
                    turn["pending"] -= 1
                log("PTT_ERROR", f"{type(e).__name__}\t{activity.log_code(activity.note('V114'))}")
                replies.put(TimedReply(dict(make_reply("", 0.0), npc=target_npc, seq=seq), job))
                return
            enqueue(job)

    tails = {}                                  # path -> LogTail; kept (parked) when the file goes quiet

    def watch():
        next_scan = 0.0
        while stop is None or not stop.is_set():
            if time.monotonic() >= next_scan:
                next_scan = time.monotonic() + WATCH_RESCAN_S
                changed, added = False, []
                recent = recent_output_logs(log_dir)
                for path in recent:
                    try:
                        if path in tails:
                            tails[path].resume()    # continues from where it stopped: nothing skipped or repeated
                        else:
                            tails[path] = LogTail(path)
                            added.append(tails[path])
                    except OSError:
                        continue
                for path, tail in tails.items():
                    if path not in recent:
                        tail.park()
                for path in [p for p in tails if not os.path.exists(p)]:
                    tail = tails.pop(path)
                    with state_lock:
                        tail.chat_epoch += 1
                        invalidate_search(tail)
                    tail.close()
                    clear_touch(path)
                    bridge.world_changed(path)
                    ptt_cancel("log_gone", path)
                    if listen["path"] == path:
                        listen_off("log_gone")
                    changed = True
                if changed or added:                # only when the set of files changes
                    names = ", ".join(os.path.basename(p) for p in sorted(tails))
                    log("WATCH", "\t".join(sorted(tails)), say="VRChat 기록 감시 중: " + (names or "(없음)"))
                    for tail in added:
                        speaker_is(tail.player)
                        world_note(tail)
                        if bridge.world_allowed(tail.world):
                            for url, _, _ in tail.playback.rows:
                                cache.watch(tv_history.youtube_id(url))
            if turn["ptt"] and time.monotonic() - turn.get("down_at", 0.0) > PTT_MAX_S:
                ptt_cancel("too_long")              # the release line never came (client closed, log lost)
            if turn["ptt"] and not mic_valid(turn["mute_epoch"]):
                ptt_cancel("osc_muted")
            busy = False
            for tail in list(tails.values()):
                for line in tail.lines():
                    busy = True
                    if listen["path"] == tail.path:
                        listen["seen"] = time.monotonic()   # that client is alive
                    handle(tail, line)
            if not busy:
                time.sleep(0.01)
        for tail in tails.values():
            with state_lock:
                invalidate_search(tail)
            tail.close()

    try:
        supervise_thread("감시", watch, stop)
    finally:
        cache.close()


def make_brain(mode: str, store=None):
    """The AI that answers, or None (= repeat what was heard) when it cannot run."""
    if mode == "off":
        return None
    if mode != "fake" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("※ API 키가 없어서 AI 없이 '들은 말' 따라 말하기로 동작합니다. "
              "이 폴더의 .env 파일에 ANTHROPIC_API_KEY=... 를 넣고 다시 켜 주세요(README 참고).", flush=True)
        return None
    try:
        return persona_ai.Brain(mode, store)
    except ImportError:
        print("※ anthropic 패키지가 없어서 AI 없이 동작합니다.", flush=True)
        return None


def main():
    persona_ai.load_env()   # .env next to this file: API key and optional settings
    ap = argparse.ArgumentParser(description="VRC AI Persona NPC companion app")
    ap.add_argument("--list", action="store_true", help="list microphones and exit")
    mic_env = os.environ.get("MIC_DEVICE", "").strip()
    ap.add_argument("--device", type=int, default=int(mic_env) if mic_env.isdigit() else None,
                    help="microphone number from --list (or MIC_DEVICE in .env)")
    ap.add_argument("--model", default=os.environ.get("WHISPER_MODEL") or "turbo",
                    help="Whisper model: turbo (default), small, medium, large-v3 ...")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--log-dir", default=VRC_DIR, help="folder with VRChat output_log_*.txt")
    ap.add_argument("--wav", default=None, help="test: use this 16-bit WAV instead of the microphone")
    ap.add_argument("--save-clips", action="store_true", help="debug: keep each recording in clips/")
    ap.add_argument("--ai", default="haiku", choices=["haiku", "sonnet", "opus", "fake", "off"],
                    help="initial model for NPCs without a saved choice: haiku (default), sonnet, opus; "
                         "fake/off keep all NPCs offline regardless of saved model")
    ap.add_argument("--open-settings", action="store_true", help="open the settings page in the browser at startup")
    ap.add_argument("--listen", default=listen_mode_from_env(), choices=LISTEN_MODES,
                    help="gaze (default): open mic while you look at the NPC from close by; ptt: hold jump only, needs a world "
                         "with jump talk switched on - off by default (or W04_LISTEN_MODE in .env)")
    args = ap.parse_args()

    if args.list:
        import sounddevice as sd
        default_in = sd.default.device[0]
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                mark = " <- 기본" if i == default_in else ""
                print(f"{i:3d}  {d['name']}  ({sd.query_hostapis(d['hostapi'])['name']}){mark}")
        return

    store = persona_store.Store(default_model=MODELS.get(args.ai, DEFAULT_MODEL))
    brain = NpcBrains(args.ai, store)
    for npc_key in persona_store.NPC_IDS:       # a data file that could not be read is said out loud once
        for name, backup in store.for_npc(npc_key).load_problems:
            log("DATA_PROBLEM", f"npc={npc_key}\tfile={name}\tbackup={backup or 'none'}",
                say=(f"주의: data/npcs/{npc_key}/{name} 파일을 읽지 못해 빈 상태로 시작합니다. "
                     + (f"원본은 같은 폴더의 {backup}로 복사해 두었습니다." if backup
                        else "원본을 복사하지 못해서 이번 실행에서는 이 파일을 덮어쓰지 않습니다.")))
    world_ids = cfg_bridge.load_world_ids(os.environ.get("W04_WORLD_IDS", ""), os.path.join(HERE, ".env.example"))
    bridge = cfg_bridge.CfgBridge(store, world_ids, log=log)
    try:                                        # before the slow Whisper load: a second copy stops at once
        server = make_server(args.port, store, brain, bridge=bridge)
    except OSError:
        log("ALREADY_RUNNING", f"port={args.port}",
            say=f"이 앱이 이미 켜져 있습니다(포트 {args.port} 사용 중). 같은 앱을 두 개 켜면 한 번 말해도 "
                "대답이 두 번 나옵니다. 이미 켜진 검은 창을 쓰거나, 그 창을 닫고 다시 켜 주세요. "
                "/ This app is already running.")
        sys.exit(1)
    stop = threading.Event()
    mute = osc_mute.MuteState()
    if not args.wav:
        mute.start(log, stop)
    print("Whisper 모델을 불러오는 중... (처음 몇 초 걸림)", flush=True)
    start_brain_warmup(brain)
    tr = Transcriber(args.model)
    rec = WavRecorder(args.wav) if args.wav else Recorder(args.device, mute=mute)
    who = ("NPC별 설정 모델" if args.ai in MODELS else
           "따라 말하기(AI 꺼짐)" if args.ai == "off" else "가짜 AI")
    how = ("NPC를 가까이서 바라보고 말하기 (월드에서 점프 말하기를 켰다면 점프로도 됨)" if args.listen == "gaze"
           else "점프 누르고 말하기 (월드의 점프 말하기 스위치가 켜져 있어야 함. 기본은 꺼짐)")
    log("START", f"port={args.port}\tmic={rec.name}\tdevice={tr.device}\tload_ms={tr.load_ms:.0f}\tai={who}\tlisten={args.listen}",
        say=f"준비 완료. 마이크: {rec.name} / 계산: {tr.device} / 답하는 쪽: {who} / 말 거는 법: {how} / "
            f"불러오기 {tr.load_ms / 1000:.1f}초")
    def watch():
        watch_log(args.log_dir, rec, tr, args.save_clips, brain, store, bridge,
                  listen_mode=args.listen, mute=mute, stop=stop)

    threading.Thread(target=supervise_thread, args=("감시", watch, stop), daemon=True).start()
    print(f"월드 연결 대기: http://127.0.0.1:{args.port}/poll  (끄려면 이 창을 닫거나 Ctrl+C)", flush=True)
    print(f"허용한 월드 {len(world_ids)}개 (Build & Test 로컬 시험 별도 허용). "
          "다른 월드에서 온 점프 표시와 설정 창 요청은 무시합니다.", flush=True)
    print(f"설정 화면(성격·플레이어 인상·기억 보기/고치기): {server.settings_url}", flush=True)
    print("  └ 이 주소는 켤 때마다 바뀝니다. settings.cmd를 더블클릭해도 열립니다. 주소를 남에게 보여 주지 마세요.", flush=True)
    try:                                        # for settings.cmd; data/ is not uploaded to git
        os.makedirs(store.root_dir, exist_ok=True)
        with open(os.path.join(store.root_dir, SETTINGS_URL_FILE), "w", encoding="utf-8") as f:
            f.write(server.settings_url)
    except OSError:
        pass
    if args.open_settings:
        webbrowser.open(server.settings_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
