"""VRC AI Persona NPC - companion app on the speaker's PC.

Hold jump in a supported world -> this app records the microphone -> release -> Whisper turns it into
text -> Claude answers as the NPC's persona (persona_ai.py, key in .env) -> the world, which has been
waiting on http://127.0.0.1:18770/poll since the press, gets the reply and the NPC says it.
Without a key, or with --ai off, the NPC just repeats what it heard ("들은 말: ...").

How it knows about the jump button: the world writes "[W04AI] PTT_DOWN" / "[W04AI] PTT_UP" into
VRChat's output_log (see README "월드 쪽 규칙"), and this app tails that file.

Run (double-click run.cmd, or):
    .venv\\Scripts\\python.exe persona_npc_app.py            default microphone
    .venv\\Scripts\\python.exe persona_npc_app.py --list     list microphones
    .venv\\Scripts\\python.exe persona_npc_app.py --device 3 use microphone #3

The microphone is kept open while the app runs; only the last 0.5 s stays in memory until you press jump.
Nothing is written to disk unless --save-clips is given.
"""
import argparse
import collections
import glob
import json
import os
import queue
import re
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import persona_ai

PORT = 18770
MARKER = "[W04AI]"                 # prefix the world puts on its output_log lines
RATE = 16000                       # Whisper works on 16 kHz mono
PRE_ROLL_S = 0.5                   # audio kept from just before the press (the log line arrives a little late)
POST_ROLL_S = 0.25                 # keep recording a moment after release (people let go on the last syllable)
MIN_CLIP_S = 0.35                  # shorter than this = accidental tap
MIN_RMS = 0.003                    # quieter than this (about -50 dBFS) = nothing was said
POLL_WAIT_S = 55
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "voice_log.tsv")
CLIP_DIR = os.path.join(HERE, "clips")
VRC_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\VRChat\VRChat")

# Whisper invents these on silent or noisy clips (its training data had many subtitled videos)
HALLUCINATIONS = re.compile(
    r"^(시청\s*해\s*주셔서\s*감사합니다|구독(과|이랑)?\s*좋아요|좋아요(와|랑)?\s*구독|자막|MBC\s*뉴스|KBS|SBS|"
    r"다음\s*영상에서\s*만나요|오늘도\s*시청)", re.IGNORECASE)

HEARD_FMT = "들은 말: {text}"
MISSED_LINE = "응? 잘 못 들었어."

_print_lock = threading.Lock()
_file_lock = threading.Lock()


def now_ms() -> float:
    return time.time() * 1000.0


def log(kind: str, detail: str = "", say: str = "") -> None:
    """One line to voice_log.tsv; `say` (Korean) is what the user sees in the console."""
    line = f"{now_ms():.0f}\t{kind}\t{detail}"
    with _file_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if say:
        with _print_lock:
            print(time.strftime("%H:%M:%S ") + say, flush=True)


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
        info = sd.query_devices(device, "input")
        self.name = info["name"]
        self.rate = RATE
        try:
            sd.check_input_settings(device=device, samplerate=RATE, channels=1, dtype="float32")
        except Exception:
            self.rate = int(info["default_samplerate"])   # resampled to 16 kHz later
        self.pre = collections.deque(maxlen=max(1, int(PRE_ROLL_S * self.rate / 512) + 1))
        self.stream = sd.InputStream(device=device, samplerate=self.rate, channels=1, dtype="float32",
                                     blocksize=512, callback=self._callback)
        self.stream.start()

    def _callback(self, indata, frames, t, status):
        block = indata[:, 0].copy()
        with self.lock:
            if self.recording:
                self.chunks.append(block)
            else:
                self.pre.append(block)

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


class WavRecorder:
    """Test stand-in for the mic: every press 'records' the same WAV file."""

    def __init__(self, path):
        self.audio = read_wav(path)
        self.name = "WAV " + os.path.basename(path)

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


# ---------------------------------------------------------------- speech to text

class Transcriber:
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

    def run(self, audio: np.ndarray):
        result = self.model.transcribe(
            audio, language="ko", task="transcribe", fp16=(self.device == "cuda"),
            temperature=0.0, condition_on_previous_text=False, without_timestamps=True,
            no_speech_threshold=0.6, logprob_threshold=-1.0)
        return result


def judge(audio: np.ndarray, held_ms: float, tr: Transcriber):
    """Returns (text or '', reason, stt_ms). held_ms = how long jump was held (from the world's log line)."""
    dur = audio.size / RATE
    if 0 <= held_ms < MIN_CLIP_S * 1000:
        return "", f"too_short held={held_ms:.0f}ms", 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < MIN_RMS:
        return "", f"too_quiet rms={rms:.4f}", 0.0
    t0 = now_ms()
    result = tr.run(audio)
    stt_ms = now_ms() - t0
    segs = result.get("segments", [])
    if segs and all(s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0 for s in segs):
        return "", "no_speech", stt_ms
    text = " ".join(result.get("text", "").split())
    if not text:
        return "", "empty", stt_ms
    if HALLUCINATIONS.search(text):
        return "", f"hallucination '{text}'", stt_ms
    return text, f"ok rms={rms:.4f} dur={dur:.2f}s", stt_ms


# ---------------------------------------------------------------- world link

replies: "queue.Queue[dict]" = queue.Queue()
_seq = 0


def make_reply(text: str, stt_ms: float) -> dict:
    return {
        "emotion": "neutral" if text else "troubled",
        "intensity": 0.5,
        "steps": [{"do": "look_at", "target": "speaker"}],
        "line": HEARD_FMT.format(text=text) if text else MISSED_LINE,
        "heard": text,
        "stt_ms": round(stt_ms),
        "seq": _seq,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        t0 = now_ms()
        path = self.path.split("?")[0]
        log("HTTP_IN", path)
        if path.startswith("/poll"):
            try:
                body = json.dumps(replies.get(timeout=POLL_WAIT_S), ensure_ascii=False)
            except queue.Empty:
                body = "{}"
            self._send(body)
        elif path.startswith("/ping"):
            self._send(f"pong {t0:.0f}")
        else:
            self._send("unknown")
        log("HTTP_OUT", f"{path}\theld_ms={now_ms() - t0:.0f}")


def latest_output_log(log_dir: str):
    files = glob.glob(os.path.join(log_dir, "output_log_*.txt"))
    return max(files, key=os.path.getmtime) if files else None


def watch_log(log_dir: str, rec, tr: Transcriber, save: bool, brain=None):
    global _seq
    current, fh = None, None
    work: "queue.Queue[tuple]" = queue.Queue()

    def worker():
        while True:
            seq, t_up, held_ms = work.get()
            audio = rec.stop()
            text, reason, stt_ms = judge(audio, held_ms, tr)
            clip = save_clip(audio, seq) if save else ""
            log("STT", f"seq={seq}\t{reason}\tstt_ms={stt_ms:.0f}\tafter_release_ms={now_ms() - t_up:.0f}\t{text}\t{clip}",
                say=(f"들음 ({stt_ms:.0f} ms): {text}" if text else f"못 알아들음 ({reason})"))
            if text and brain is not None:
                ai = brain.reply(text)
                reply = {k: ai[k] for k in ("emotion", "intensity", "steps", "lines", "line")}
                reply.update(heard=text, stt_ms=round(stt_ms), ai_ms=ai["ai_ms"], seq=_seq)
                err = ai.get("error", "")
                log("AI", f"seq={seq}\tmodel={ai['model']}\tai_ms={ai['ai_ms']}\t{ai.get('usage', '')}\t{err}\t"
                          f"{reply['emotion']}\t{json.dumps(reply['steps'], ensure_ascii=False)}\t{reply['line']}",
                    say=(f"AI 오류 ({err})" if err else
                         f"kwonjun ({ai['ai_ms'] / 1000:.1f}초, {reply['emotion']}): {reply['line']}"))
            else:
                reply = make_reply(text, stt_ms)
            replies.put(reply)
            log("REPLY", f"seq={seq}\tafter_release_ms={now_ms() - t_up:.0f}")

    threading.Thread(target=worker, daemon=True).start()
    while True:
        newest = latest_output_log(log_dir)
        if newest and newest != current:
            if fh:
                fh.close()
            current = newest
            fh = open(current, "r", encoding="utf-8", errors="replace")
            fh.seek(0, os.SEEK_END)
            log("WATCH", current, say="VRChat 기록 감시 중: " + os.path.basename(current))
        line = fh.readline() if fh else ""
        if not line:
            time.sleep(0.01)
            continue
        if MARKER not in line:
            continue
        if "PTT_DOWN" in line:
            while not replies.empty():          # a new utterance: drop replies nobody picked up
                replies.get_nowait()
            rec.start()
            log("PTT_DOWN", line.strip()[-80:], say="녹음 시작")
        elif "PTT_UP" in line:
            _seq += 1
            m = re.search(r"held_ms=(\d+)", line)
            held_ms = float(m.group(1)) if m else -1.0
            log("PTT_UP", line.strip()[-80:], say="녹음 끝, 글자로 바꾸는 중...")
            work.put((_seq, now_ms(), held_ms))


def make_brain(mode: str):
    """The AI that answers, or None (= repeat what was heard) when it cannot run."""
    if mode == "off":
        return None
    if mode != "fake" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("※ API 키가 없어서 AI 없이 '들은 말' 따라 말하기로 동작합니다. "
              "이 폴더의 .env 파일에 ANTHROPIC_API_KEY=... 를 넣고 다시 켜 주세요(README 참고).", flush=True)
        return None
    try:
        return persona_ai.Brain(mode)
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
    ap.add_argument("--ai", default="haiku", choices=["haiku", "sonnet", "fake", "off"],
                    help="who answers: Claude Haiku (default), Claude Sonnet, a fake AI for tests, or off (repeat what was heard)")
    args = ap.parse_args()

    if args.list:
        import sounddevice as sd
        default_in = sd.default.device[0]
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                mark = " <- 기본" if i == default_in else ""
                print(f"{i:3d}  {d['name']}  ({sd.query_hostapis(d['hostapi'])['name']}){mark}")
        return

    brain = make_brain(args.ai)
    print("Whisper 모델을 불러오는 중... (처음 몇 초 걸림)", flush=True)
    tr = Transcriber(args.model)
    rec = WavRecorder(args.wav) if args.wav else Recorder(args.device)
    who = "따라 말하기(AI 꺼짐)" if brain is None else f"AI {brain.model}"
    log("START", f"port={args.port}\tmic={rec.name}\tdevice={tr.device}\tload_ms={tr.load_ms:.0f}\tai={who}",
        say=f"준비 완료. 마이크: {rec.name} / 계산: {tr.device} / 답하는 쪽: {who} / 불러오기 {tr.load_ms / 1000:.1f}초")
    threading.Thread(target=watch_log, args=(args.log_dir, rec, tr, args.save_clips, brain), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"월드 연결 대기: http://127.0.0.1:{args.port}/poll  (끄려면 이 창을 닫거나 Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
