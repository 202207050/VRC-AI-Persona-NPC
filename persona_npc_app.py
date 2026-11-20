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
    just before), ends it after 1.3 s of silence by default or 15 s, and drops blips under 0.35 s of speech.
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
[W04AI] lines count only from a log whose current world is allowed (Build & Test, or W04_WORLD_IDS in .env).

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
import sys
import threading
import time
import wave
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import numpy as np

import cfg_bridge
from voice_options import DEFAULT_MODEL, MODELS, hint_prompt, parse_hints, replace_hints
from datetime import datetime

import persona_ai
import persona_store
import smalltalk

PORT = 18770
MARKER = "[W04AI]"                 # prefix the world puts on its output_log lines
RATE = 16000                       # Whisper works on 16 kHz mono
PRE_ROLL_S = 0.5                   # audio kept from just before the press (the log line arrives a little late)
POST_ROLL_S = 0.25                 # keep recording a moment after release (people let go on the last syllable)
MIN_CLIP_S = 0.35                  # shorter than this = accidental tap
MIN_RMS = 0.003                    # quieter than this (about -50 dBFS) = nothing was said
POLL_WAIT_S = 55
PTT_MAX_S = 30.0                   # a held jump is dropped after this long (the PTT_UP line was lost)
WORLD_MAX_LINE = 100               # the world drops a reply whose line is longer (W04Talk.MaxLine)
LISTEN_MODES = ("gaze", "ptt")     # gaze: open mic while the world says LISTEN_ON; ptt: hold jump only
VAD_FRAME_S = 0.03                 # the detector judges the microphone in 30 ms steps
VAD_ONSET_FRAMES = 8               # speech starts when 5 of the last 8 steps (0.24 s) are loud
VAD_ONSET_VOICED = 5
VAD_END_SILENCE_S = 1.3            # default quiet time before ending the utterance
FRAGMENT_GAP_S = 2.0               # measured from the last voiced frame to the next speech onset
VAD_MAX_S = 15.0                   # an utterance is cut at this length
VAD_MIN_SPEECH_S = MIN_CLIP_S      # less loud time than this = a blip (cough, click), dropped
VAD_TAIL_S = 0.25                  # quiet kept after the last loud step (the rest is trimmed for Whisper)
VAD_MIN_THRESHOLD = 0.006          # never call anything quieter than this speech (about -44 dBFS)
VAD_FLOOR_RATIO = 3.5              # loud = this many times the room's noise floor
LISTEN_STALE_S = 120               # LISTEN_ON ends by itself if that VRChat log writes nothing this long
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "voice_log.tsv")
CLIP_DIR = os.path.join(HERE, "clips")
VRC_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\VRChat\VRChat")
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
EVENT_RE = re.compile(r"\[W04AI\] (PTT_DOWN|PTT_UP|LISTEN_ON|LISTEN_OFF|NPC_STATE|NPC_CHAT_REQ|AFFECTION|NPC_POSE|GLOBAL|CONTENT_LEVEL)\b")
CONTENT_LEVEL_RE = re.compile(cfg_bridge._STAMP + r"\[W04AI\] CONTENT_LEVEL value=([0-2])\s*$")
GLOBAL_RE = re.compile(cfg_bridge._STAMP + r"\[W04AI\] GLOBAL on=([01])\s*$")
AFFECTION_RE = re.compile(r"\[W04AI\] AFFECTION type=(pat|hug|wave) npc=(\S+) by=(.+?)\s*$")
POSE_RE = re.compile(r"\[W04AI\] NPC_POSE npc=(\S+) pose=(stand|sit|lie)\b")
NPC_ADDRESS = {
    "kwonjun": re.compile(r"권\s*[준쥰]|(?<![a-z])kwon[\s-]*jung?(?![a-z])", re.IGNORECASE),
    "itoa": re.compile(r"이\s*토\s*[아알]|(?<![a-z])i[\s-]*to[\s-]*ar?(?![a-z])", re.IGNORECASE),
}
FOLLOW_UP_MS = 20_000
DEFAULT_NPC_STATE = {"kwonjun": True, "itoa": False}


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
        self.lock = threading.RLock()

    def for_npc(self, npc=None):
        npc = persona_store.npc_id(npc)
        with self.lock:
            if npc not in self.brains:
                self.brains[npc] = make_brain(self.mode, self.store.for_npc(npc))
            return self.brains[npc]


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


def log(kind: str, detail: str = "", say: str = "") -> None:
    """One line to voice_log.tsv; `say` (Korean) is what the user sees in the console."""
    line = f"{now_ms():.0f}\t{kind}\t{detail}"
    with _file_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if say:
        with _print_lock:
            print(time.strftime("%H:%M:%S ") + say, flush=True)


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


# ---------------------------------------------------------------- microphone

class Recorder:
    """Keeps the mic open; a short ring buffer covers the moment before the press."""

    def __init__(self, device=None):
        import sounddevice as sd
        self.sd = sd
        self.device = device
        self.lock = threading.Lock()
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
        block = indata[:, 0].copy()
        with self.lock:
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
            for block in self.pre:
                self.tap.put(block)
            return self.tap

    def close_tap(self):
        with self.lock:
            self.tap = None

    def start(self):
        with self.lock:
            self.chunks = list(self.pre)
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

    def _frame(self, f: np.ndarray):
        rms = float(np.sqrt(np.mean(f * f)))
        up = 0.001 if self.active else 0.003
        self.floor = max(0.0005, self.floor + (rms - self.floor) * (0.2 if rms < self.floor else up))
        threshold = max(VAD_MIN_THRESHOLD, self.floor * VAD_FLOOR_RATIO)
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
SENTENCE_END = re.compile(
    r'''[.!?。！？…]["'”’」』)]*$|'''
    r"(?:니다|니까|어요|아요|세요|해요|예요|네요|군요|겠다|한다|했다|이다|더라|잖아|줄래|할래|거야|구나|했어)$")


def collapse_transcript_runs(text):
    return TRANSCRIPT_RUN.sub(lambda m: (m[1] or m[2]) * (3 if m[1] or len(m[2]) == 1 else 2), text)


def is_fragment(text):
    return len("".join(text.split())) < 4 or not SENTENCE_END.search(text)

class Transcriber:
    supports_initial_prompt = True

    def __init__(self, model_name: str):
        import torch
        import whisper
        self.whisper = whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        result = self.model.transcribe(
            audio, language="ko", task="transcribe", fp16=(self.device == "cuda"),
            temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
            no_speech_threshold=0.6, logprob_threshold=-1.0, initial_prompt=initial_prompt)
        return result


def judge(audio: np.ndarray, held_ms: float, tr: Transcriber, hints: str = "", raw_sink=None):
    """Returns (text or '', reason, stt_ms). held_ms = how long jump was held (from the world's log line)."""
    dur = audio.size / RATE
    if 0 <= held_ms < MIN_CLIP_S * 1000:
        return "", f"too_short held={held_ms:.0f}ms", 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < MIN_RMS:
        return "", f"too_quiet rms={rms:.4f}", 0.0
    t0 = now_ms()
    pairs = parse_hints(hints)
    prompt = hint_prompt(pairs)
    # Older/offline transcribers can opt out; replacement still works without prompt support.
    result = (tr.run(audio, initial_prompt=prompt) if prompt and getattr(tr, "supports_initial_prompt", False)
              else tr.run(audio))
    stt_ms = now_ms() - t0
    segs = result.get("segments", [])
    if segs and all(s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0 for s in segs):
        return "", "no_speech", stt_ms
    raw_text = result.get("text", "")
    if raw_sink is not None:
        raw_sink.append(raw_text)
    text = collapse_transcript_runs(" ".join(raw_text.split()))
    if not text:
        return "", "empty", stt_ms
    if HALLUCINATIONS.search(text):
        return "", f"hallucination '{text}'", stt_ms
    return collapse_transcript_runs(replace_hints(text, pairs)), f"ok rms={rms:.4f} dur={dur:.2f}s", stt_ms


# ---------------------------------------------------------------- world link

class SearchReply(dict):
    def __init__(self, payload, job, final=False):
        super().__init__(payload)
        self.tail, self.npc_epoch, self.final = job.tail, job.npc_epoch, final

    def valid(self):
        npc = self["npc"]
        return self.tail.npc_epoch[npc] == self.npc_epoch and self.tail.npc_state[npc]


class ReplyQueue(queue.Queue):
    # W04Probe keeps one pending body per NPC for 11 s. A later body could replace a search result.
    search_play_window = 11.5

    def __init__(self):
        super().__init__()
        self.protected_until = {}

    def discard(self, keep):
        with self.mutex:
            self.queue = collections.deque(reply for reply in self.queue if keep(reply))

    def get_for_world(self, timeout):
        deadline = time.monotonic() + timeout
        with self.not_empty:
            while True:
                now = time.monotonic()
                wake = deadline
                for i, reply in enumerate(self.queue):
                    if isinstance(reply, SearchReply) and not reply.valid():
                        continue
                    npc = reply.get("npc")
                    ready = self.protected_until.get(npc, (None, 0))[1]
                    if ready > now:
                        wake = min(wake, ready)
                        continue
                    del self.queue[i]
                    if isinstance(reply, SearchReply) and reply.final:
                        self.protected_until[npc] = reply["seq"], now + self.search_play_window
                    return reply
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

    def context(self):
        return self.tail, self.npc, self.npc_epoch, self.speaker


def make_empty_reply(target: str) -> dict:
    """Answer for a push-to-talk that was dictated into the settings panel: no "line", so the world
    treats it like the 55 s timeout answer {} (POLL_EMPTY) and just closes the thinking dots."""
    return {"line": "", "dictate": target, "seq": _seq}


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


class Handler(BaseHTTPRequestHandler):
    """/poll and /ping are for the world (GET only, no token: unchanged).
    /cfg is for the world's settings panel: no token, open only shortly after the panel says CFG_OPEN.
    /settings and /api/* are for the user's browser and need the per-run token."""

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: str, code: int = 200, ctype: str = "text/plain") -> None:
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
        self.end_headers()
        self.wfile.write(data)

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
        if path in ("/poll", "/ping") or path.startswith("/ping/"):
            # The world asks with this PC's own address and never sends Origin. Another site's name pointed
            # at 127.0.0.1 (DNS rebinding) or a web page's cross-site request is turned away BEFORE a reply
            # is taken from the queue, so it can neither read nor swallow an answer.
            if ((self.headers.get("Host") or "").lower() not in self._local_names()
                    or self.headers.get("Origin") is not None
                    or (self.headers.get("Sec-Fetch-Site") or "none").lower() not in ("none", "same-origin")):
                self._send("{}", 403, "application/json")
            elif path == "/poll":
                try:
                    reply = replies.get_for_world(POLL_WAIT_S)
                except queue.Empty:
                    self._send("{}")
                else:
                    try:
                        self._send(json.dumps(reply, ensure_ascii=False))
                    except OSError:
                        replies.retry_search(reply)
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

    What is already in the file when it is first seen is read once for the world and the player name
    only; [W04AI] lines in that part are old and are never acted on."""

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
        self.buf = b""
        self.pos = 0
        self.fh = open(path, "rb")
        end = os.path.getsize(path)
        while self.fh.tell() < end:
            raw = self.fh.readline()
            if not raw.endswith(b"\n"):                 # half-written last line: read it again later, live
                self.fh.seek(-len(raw), os.SEEK_CUR)
                break
            line = raw.decode("utf-8", errors="replace")
            world = cfg_bridge.world_from_line(line, self.world)
            if world != self.world:
                self.npc_state = dict(DEFAULT_NPC_STATE)
            self.world = world
            update_npc_state(self.npc_state, line)
            if "Initialized PlayerAPI" in line:
                m = LOCAL_PLAYER_RE.search(line)
                if m:
                    self.player = m.group(1)

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
              listen_mode: str = None):
    """Follows every recent output_log at once. [W04AI] lines count only from a log whose client is in
    an allowed world (bridge.world_allowed). stop: a threading.Event that ends the loop (tests).
    listen_mode: "gaze" (LISTEN_ON/OFF open the mic to the voice detector; jump still works) or "ptt"."""
    global _seq
    store = store or (brain.store if brain is not None else persona_store.Store())
    bridge = bridge or cfg_bridge.CfgBridge(store, log=log)
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
    chat_inflight = False
    chats = None

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
        if name and name != store.speaker:
            store.set_speaker(name)
            log("SPEAKER", name, say=f"말하는 사람(내 VRChat 이름): {name}")

    work: "queue.Queue[Utterance]" = queue.Queue()

    def set_stage(stage):
        with state_lock:
            turn["stage"] = stage

    def done():
        with state_lock:
            turn["pending"] -= 1

    def failed(job, error):
        try:
            log("WORKER_ERROR", f"seq={job.seq}\t{type(error).__name__}: {str(error)[:120]}",
                say=f"이번 말은 처리하지 못함 ({type(error).__name__}). 다음 말부터 다시 받습니다.")
            if not job.by_voice:
                replies.put(dict(make_reply("", 0.0), npc=job.npc, seq=job.seq))
        except OSError:
            pass

    def complete(job, prepared):
        set_stage("answer")
        try:
            answer(job, *prepared)
        except Exception as error:
            failed(job, error)
        finally:
            done()

    def worker():
        fragment = None
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
                        complete(previous, prepared)
                        fragment = None
                continue

            if fragment:
                previous, prior = fragment
                if (not job.by_voice or job.dictate is not None or job.context() != previous.context()
                        or job.started > previous.ended + FRAGMENT_GAP_S):
                    complete(previous, prior)
                    fragment = None
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
                _, prior = fragment
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
        dictate = job.dictate
        if dictate is UNBOUND:              # push-to-talk keeps its old rule: what is armed when it is handled
            dictate = bridge.dictate_target()
        elif dictate is not None and bridge.dictate_target() != dictate:
            # it was recorded FOR the panel and that dictation was cancelled, replaced or timed out while the
            # job waited: thrown away. It must never fall through to the AI as ordinary talk.
            log("STT", f"seq={job.seq}\tdictate_gone\tafter_release_ms={now_ms() - job.t_up:.0f}",
                say="받아쓰기가 취소됐거나 시간이 지나서 이번 말은 버림")
            return
        hint_store = store.for_npc(dictate.get("npc") if dictate else job.npc)
        with hint_store.lock:
            hints = hint_store.settings["stt_hints"]
        raw = []
        text, reason, stt_ms = judge(job.audio, job.held_ms, tr, hints, raw_sink=raw)
        clip = save_clip(job.audio, job.seq) if save else ""
        return text, reason, stt_ms, clip, dictate, raw[0] if raw else ""

    def answer(job, text, reason, stt_ms, clip, dictate, raw):
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
                replies.put(dict(make_empty_reply(dictate["target"]), npc=npc))
                log("REPLY", f"seq={seq}\tafter_release_ms={now_ms() - t_up:.0f}\tdictate")
            return
        log("STT", f"seq={seq}\t{reason}\tstt_ms={stt_ms:.0f}\tafter_release_ms={now_ms() - t_up:.0f}\t{speaker}\t{text}\t{clip}",
            say=(f"{speaker} 마이크 ({stt_ms:.0f} ms): {text}" if text else f"못 알아들음 ({reason})"))
        selected_brain = select_brain(brain, npc) if text else None
        searching = False

        def search_valid():
            return (job.tail.npc_epoch[npc] == job.npc_epoch and job.tail.npc_state[npc]
                    and (stop is None or not stop.is_set()))

        if text and selected_brain is not None:
            def interim(r):
                # web search turn: show "잠깐만, 찾아볼게!" now; "more" makes the world poll again for the answer
                nonlocal searching
                searching = r.get("interim") == "search"
                first = {k: r[k] for k in ("emotion", "intensity", "steps", "lines", "line") if k in r}
                first["steps"] = [step for step in first["steps"] if step["do"] in ("look_at", "gesture", "wait")]
                first.update(heard=text, stt_ms=round(stt_ms), seq=seq, more=True, npc=npc)
                with state_lock:
                    if searching:
                        if not search_valid():
                            return
                        turn["stage"] = "search"
                    replies.put(SearchReply(first, job) if searching else first)
                doing = "기억 되짚는 중" if r.get("interim") == "recall" else "검색 중"
                log("REPLY", f"seq={seq}\tinterim\t{first.get('line', '')}", say=f"{st.npc_name()}: {first.get('line', '')} ({doing})")

            options = {"global_mode": True} if job.tail.global_mode else {}
            if isinstance(selected_brain, persona_ai.Brain):
                captured_at = time.time() - max(0, now_ms() - t_up) / 1000
                last_epoch, last_speaker, last_at = job.tail.last_answer.get(npc, (-1, "", 0))
                follow_up = (last_epoch == job.npc_epoch and last_speaker == speaker
                             and 0 <= t_up - last_at <= FOLLOW_UP_MS)
                named = NPC_ADDRESS[npc].search(text) is not None
                display_name = st.npc_name()
                if not named and display_name != persona_store.DEFAULT_NPC_NAMES[npc]:
                    pattern = r"(?<![a-z])" + re.escape(display_name) + r"(?![a-z])"
                    named = bool(display_name and re.search(pattern, text, re.IGNORECASE))
                with st.lock:
                    check_address = st.settings["filter_overheard"] and not (follow_up or named)
                options.update(content_level=job.tail.content_level, utterance={
                    "speaker": speaker, "source": "PC 주인 마이크", "raw": raw,
                    "time": datetime.fromtimestamp(captured_at).astimezone().isoformat(timespec="seconds"),
                }, check_address=check_address)
            ai = selected_brain.reply(text, speaker=speaker, on_interim=interim, **options)
            if ai.get("for_me") is False:
                log("AI_SKIP", f"seq={seq}\tnpc={npc}\tai_ms={ai['ai_ms']}\t넘김",
                    say=f"{st.npc_name()}: 다른 사람에게 한 말로 판단해 넘김")
                if not by_voice and search_valid():
                    replies.put({"npc": npc, "seq": seq, "line": "", "steps": []})
                return
            reply = {k: ai[k] for k in ("emotion", "intensity", "steps", "lines", "line")}
            # The world uses the live pose when it starts a movement step, including getting up.
            reply.update(heard=text, stt_ms=round(stt_ms), ai_ms=ai["ai_ms"], seq=seq, npc=npc)
            err = ai.get("error", "")
            kept = ai.get("remembered") or {}
            if ai.get("recall"):
                log("RECALL", f"seq={seq}\trecall_ms={ai.get('recall_ms', 0)}")
            log("AI", f"seq={seq}\tmodel={ai['model']}\tai_ms={ai['ai_ms']}\t{ai.get('usage', '')}\t{err}\t"
                      f"{reply['emotion']}\t{json.dumps(reply['steps'], ensure_ascii=False)}\t{reply['line']}",
                say=(f"AI 오류 ({err})" + ai_error_hint(err) if err else
                     f"{st.npc_name()} ({ai['ai_ms'] / 1000:.1f}초, {reply['emotion']}): {reply['line']}"))
            if kept.get("impression"):
                log("IMPRESSION", f"seq={seq}\t{speaker}\t{kept['impression']}",
                    say=f"  └ {speaker}에 대한 인상을 고쳐 씀: {kept['impression']}")
            if kept.get("memory_note"):
                log("MEMORY", f"seq={seq}\t{kept['memory_note']}", say=f"  └ 기억에 기록함: {kept['memory_note']}")
        elif by_voice and not text:
            # open mic: a noise that was not speech must not make the NPC say "응? 잘 못 들었어."
            return
        else:
            reply = make_reply(text, stt_ms)
        reply.update(npc=npc, seq=seq)
        with state_lock:
            if searching:
                if not search_valid():
                    return
                if turn["pending"] > 1 or turn["gaze_active"] or turn["ptt"]:
                    reply["more"] = True
                replies.put(SearchReply(reply, job, final=True))
            else:
                replies.put(reply)
            if selected_brain is not None and reply["line"] and not ai.get("error") and search_valid():
                job.tail.last_answer[npc] = (job.npc_epoch, speaker, now_ms())
        log("REPLY", f"seq={seq}\tafter_release_ms={now_ms() - t_up:.0f}")
        if selected_brain is not None:      # the reply is already on its way: tidy the memos if they grew
            try:
                start_compaction(st, selected_brain)
            except Exception as e:
                log("COMPACT", f"npc={npc}\tok=0\t{type(e).__name__}")

    threading.Thread(target=worker, daemon=True).start()

    def listener():
        """Gaze mode: while LISTEN is on (or a dictation is armed) the microphone goes through the
        voice detector. The tap closes as soon as listening ends and no utterance is in progress, so
        nothing is looked at outside those times."""
        global _seq
        det = UtteranceDetector(getattr(rec, "rate", RATE))
        tap = None
        utterance_tail = None
        utterance_npc, utterance_speaker, utterance_dictate = "kwonjun", "", None
        utterance_epoch, utterance_started = 0, 0.0
        while stop is None or not stop.is_set():
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
                with state_lock:
                    continuing = det.active
                    blocked = turn["pending"] and turn["stage"] == "answer"
                    if turn["ptt"] or (blocked and not continuing):
                        det.reset()  # speech during a player reply is dropped
                    else:
                        was_active = det.active
                        if not was_active:      # fixed when the speech starts, carried by the job
                            utterance_npc, utterance_speaker, utterance_dictate = current_npc, current_speaker, armed
                            utterance_tail = current_tail
                            utterance_epoch = current_tail.npc_epoch[current_npc] if current_tail else 0
                        finished = det.feed(block)
                        if not was_active and (det.active or finished):
                            turn["speech_epoch"] += 1
                            utterance_started = time.monotonic() - VAD_ONSET_VOICED * VAD_FRAME_S
                        for audio, speech_s in finished:
                            if speech_s < VAD_MIN_SPEECH_S:
                                blips.append(speech_s)
                                continue
                            if (utterance_dictate is None and
                                    (utterance_tail is None or not utterance_tail.npc_state[utterance_npc]
                                     or utterance_epoch != utterance_tail.npc_epoch[utterance_npc])):
                                continue
                            clear_replies(preserve_search=True)
                            _seq += 1
                            turn["pending"] += 1
                            ended = time.monotonic() - det.trailing_silence
                            log("VAD_END", f"seq={_seq}\tspeech_s={speech_s:.2f}\tclip_s={audio.size / det.rate:.2f}",
                                say="말 끝남, 글자로 바꾸는 중...")
                            work.put(Utterance(_seq, now_ms(), -1.0, to_16k(audio, det.rate), utterance_npc, True,
                                              utterance_tail, utterance_epoch, utterance_speaker, utterance_dictate,
                                              utterance_started, ended))
                            det.reset()
                            break
                    context = (utterance_tail, utterance_npc, utterance_epoch, utterance_speaker)
                    turn["gaze_active"] = (context, utterance_started, utterance_dictate) if det.active else None
                for speech_s in blips:
                    log("VAD_BLIP", f"speech_s={speech_s:.2f}")
            if not want and not det.active:
                rec.close_tap()
                tap = None
                det.reset()
        if tap is not None:
            rec.close_tap()

    if gaze:
        threading.Thread(target=listener, daemon=True).start()

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
        if hasattr(rec, "cancel"):
            rec.cancel()
        else:
            rec.stop()
        log("PTT_CANCEL", reason, say="녹음 취소 (월드를 떠났거나 점프 해제 기록이 없음)")

    def invalidate_search(tail, npc=None):
        # Called with state_lock, so a search cannot publish after its NPC was turned off.
        for key in (tail.npc_epoch if npc is None else (npc,)):
            tail.npc_epoch[key] += 1
        replies.discard(lambda reply: not (isinstance(reply, SearchReply) and reply.tail is tail
                                          and (npc is None or reply["npc"] == npc)))

    def handle(tail: LogTail, line: str):
        global _seq
        if MARKER not in line:
            world = cfg_bridge.world_from_line(line, tail.world)
            if world != tail.world or ("Joining " in line and cfg_bridge.JOIN_RE.match(line)):
                tail.global_mode = False
                tail.content_level = 0
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
        if bridge.handle_line(tail.path, tail.world, line):
            return
        m = EVENT_RE.search(line)
        if not m:
            return
        event = m.group(1)
        npc = event_npc(line)
        if not bridge.world_allowed(tail.world):
            if event == "PTT_DOWN":
                log("PTT_IGNORED", "world_not_allowed", say="점프 표시를 무시함 (허용하지 않은 월드)")
            elif event == "LISTEN_ON":
                log("LISTEN_IGNORED", "world_not_allowed", say="듣기 표시를 무시함 (허용하지 않은 월드)")
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
        if event == "AFFECTION":
            m = AFFECTION_RE.search(line)
            if m:
                kind, who = m.group(1), m.group(3)[:persona_store.MAX_NAME_CHARS]
                st = store.for_npc(npc)
                phrase = st.add_affection(who, kind)
                log("AFFECTION", f"type={kind}\tnpc={npc}\tby={who}\tnoted={int(bool(phrase))}",
                    say=f"  └ {st.npc_name()}: {who}에 대한 인상에 기록함: {phrase}" if phrase else "")
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
            log("LISTEN_ON", line.strip()[-80:], say="NPC가 듣는 중 (그냥 말하면 됩니다)")
        elif event == "LISTEN_OFF":
            if listen["path"] == tail.path and listen["npc"] == npc:
                listen_off(line.strip()[-80:], say="듣기 멈춤 (NPC를 안 보거나 멀어짐)")
        elif event == "PTT_DOWN":
            with state_lock:
                clear_replies(preserve_search=True)
                turn["speech_epoch"] += 1
                turn["ptt"] = True
                turn["npc"] = npc
                turn["npc_epoch"] = tail.npc_epoch[npc]
                turn["path"] = tail.path
                turn["down_at"] = time.monotonic()
            rec.start()
            log("PTT_DOWN", line.strip()[-80:], say="녹음 시작")
        else:
            # Ignore unmatched releases (including presses refused while this NPC was off).
            if not turn["ptt"] or turn.get("path") != tail.path:
                return
            m = re.search(r"held_ms=(\d+)", line)
            held_ms = float(m.group(1)) if m else -1.0
            log("PTT_UP", line.strip()[-80:], say="녹음 끝, 글자로 바꾸는 중...")
            with state_lock:
                target_npc = turn["npc"]
                npc_epoch = turn["npc_epoch"]
                _seq += 1
                seq = _seq
                turn["pending"] += 1
                turn["ptt"] = False
            # the recording is taken off the microphone right here (0.25 s post-roll), so a second press
            # while this one is still being answered can never swap its audio or its NPC
            t_up = now_ms()
            try:
                audio = rec.stop()
            except Exception as e:
                with state_lock:
                    turn["pending"] -= 1
                log("PTT_ERROR", type(e).__name__)
                replies.put(dict(make_reply("", 0.0), npc=target_npc, seq=seq))
                return
            work.put(Utterance(seq, t_up, held_ms, audio, target_npc, False, tail, npc_epoch, speaker_of(tail)))

    tails = {}                                  # path -> LogTail; kept (parked) when the file goes quiet
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
        if turn["ptt"] and time.monotonic() - turn.get("down_at", 0.0) > PTT_MAX_S:
            ptt_cancel("too_long")              # the release line never came (client closed, log lost)
        busy = False
        for tail in list(tails.values()):
            try:
                for line in tail.lines():
                    busy = True
                    if listen["path"] == tail.path:
                        listen["seen"] = time.monotonic()   # that client is alive
                    handle(tail, line)
            except OSError:
                pass
        if not busy:
            time.sleep(0.01)
    for tail in tails.values():
        tail.close()


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
    world_ids = cfg_bridge.parse_world_ids(os.environ.get("W04_WORLD_IDS", ""))
    bridge = cfg_bridge.CfgBridge(store, world_ids, log=log)
    try:                                        # before the slow Whisper load: a second copy stops at once
        server = make_server(args.port, store, brain, bridge=bridge)
    except OSError:
        log("ALREADY_RUNNING", f"port={args.port}",
            say=f"이 앱이 이미 켜져 있습니다(포트 {args.port} 사용 중). 같은 앱을 두 개 켜면 한 번 말해도 "
                "대답이 두 번 나옵니다. 이미 켜진 검은 창을 쓰거나, 그 창을 닫고 다시 켜 주세요. "
                "/ This app is already running.")
        sys.exit(1)
    print("Whisper 모델을 불러오는 중... (처음 몇 초 걸림)", flush=True)
    tr = Transcriber(args.model)
    rec = WavRecorder(args.wav) if args.wav else Recorder(args.device)
    who = ("NPC별 설정 모델" if args.ai in MODELS else
           "따라 말하기(AI 꺼짐)" if args.ai == "off" else "가짜 AI")
    how = ("NPC를 가까이서 바라보고 말하기 (월드에서 점프 말하기를 켰다면 점프로도 됨)" if args.listen == "gaze"
           else "점프 누르고 말하기 (월드의 점프 말하기 스위치가 켜져 있어야 함. 기본은 꺼짐)")
    log("START", f"port={args.port}\tmic={rec.name}\tdevice={tr.device}\tload_ms={tr.load_ms:.0f}\tai={who}\tlisten={args.listen}",
        say=f"준비 완료. 마이크: {rec.name} / 계산: {tr.device} / 답하는 쪽: {who} / 말 거는 법: {how} / "
            f"불러오기 {tr.load_ms / 1000:.1f}초")
    threading.Thread(target=watch_log, args=(args.log_dir, rec, tr, args.save_clips, brain, store, bridge),
                     kwargs={"listen_mode": args.listen}, daemon=True).start()
    print(f"월드 연결 대기: http://127.0.0.1:{args.port}/poll  (끄려면 이 창을 닫거나 Ctrl+C)", flush=True)
    print(f"허용한 월드: Build & Test(로컬 시험) + .env의 W04_WORLD_IDS {len(world_ids)}개. "
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


if __name__ == "__main__":
    main()
