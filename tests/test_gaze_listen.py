"""Offline check of gaze-gated listening (LISTEN_ON / LISTEN_OFF + the voice detector) and of ptt mode.

    .venv\\Scripts\\python.exe tests\\test_gaze_listen.py

No microphone, no Whisper, no API call, no real VRChat log: a fake microphone is fed synthetic sound
(loud noise = "speech", faint noise = a quiet room), fake logs live in a temporary folder, and
speech-to-text and the Claude client are fakes.
"""
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import wave
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)

import cfg_bridge              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402

checks = 0
LOCAL = "local:1a7a51216f56f32584745f9ce2887998a0e3a76e83f7e20749da576bb20137da"
OTHER = "wrld_b05ed97c-0856-4c72-bda6-c1a99bae30be"
T = "2026.11.19 13:44:37 Debug      -  "
LISTEN_ON = T + "[W04AI] LISTEN_ON ang=6 dist=1.4 utc=1\n"
LISTEN_OFF = T + "[W04AI] LISTEN_OFF utc=2\n"
PTT = T + "[W04AI] PTT_DOWN utc=1\n" + T + "[W04AI] PTT_UP held_ms=1500 utc=2\n"
RNG = np.random.default_rng(7)


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def join(world):
    tail = "" if world.startswith("local:") else ":59823~private(usr_86bc47ca-4221-47cf-b5cd-b2341fd4c964)~region(jp)"
    return (T + "[Behaviour] Entering Room: Some Room\n" + T + f"[Behaviour] Joining {world}{tail}\n"
            + T + "[Behaviour] Joining or Creating Room: Some Room\n")


def wait_for(cond, what, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out: " + what)


def speech(seconds, rate=app.RATE, level=0.1):
    return (RNG.standard_normal(int(seconds * rate)) * level).astype(np.float32)


def room(seconds, rate=app.RATE):
    return speech(seconds, rate, 0.001)


class FakeMic:
    """A microphone the test feeds by hand. Sound pushed while nobody listens is counted, not kept."""
    name = "fake mic"
    rate = app.RATE

    def __init__(self):
        self.tap = None
        self.unheard = 0

    def open_tap(self):
        self.tap = queue.Queue()
        return self.tap

    def close_tap(self):
        self.tap = None

    def push(self, audio):
        tap = self.tap
        if tap is None:
            self.unheard += audio.size
            return
        for i in range(0, audio.size, 512):
            tap.put(audio[i:i + 512])

    def drained(self):
        return self.tap is None or self.tap.empty()

    def start(self):
        pass

    def stop(self):
        return speech(1.0)


class FakeTr:
    """Stands in for Whisper: every clip 'says' self.text; remembers how long each clip was."""
    text = "안녕"

    def __init__(self):
        self.clips = []

    def run(self, audio):
        self.clips.append(audio.size / app.RATE)
        return {"text": self.text, "segments": []}


class SlowBrain:
    """Stands in for the AI; `gate` holds an answer back until the test lets it go."""

    def __init__(self, store):
        self.store, self.model, self.heard = store, "mock", []
        self.gate = threading.Event()
        self.gate.set()

    def reply(self, text, **kw):
        self.heard.append(text)
        self.gate.wait(10)
        return {"emotion": "happy", "intensity": 0.5, "steps": [], "lines": {"ko": "응!"}, "line": "응! " + text,
                "ai_ms": 1, "model": "mock"}


def detector_checks(tmp):
    def run(audio, rate=app.RATE, block=512):
        det = app.UtteranceDetector(rate)
        out = []
        for i in range(0, audio.size, block):
            out += det.feed(audio[i:i + block])
        return out, det

    # ---- speech then silence = one utterance, with the pre-roll and a short tail
    out, det = run(np.concatenate([room(1.0), speech(1.5), room(1.4)]))
    ok(len(out) == 1, f"one utterance (got {len(out)})")
    audio, speech_s = out[0]
    ok(1.3 <= speech_s <= 1.7, f"loud time is about the 1.5 s spoken ({speech_s:.2f})")
    ok(1.5 + 0.3 <= audio.size / app.RATE <= 1.5 + app.PRE_ROLL_S + app.VAD_TAIL_S + 0.2,
       f"clip = pre-roll + speech + short tail ({audio.size / app.RATE:.2f} s)")
    lead = audio[:int(0.2 * app.RATE)]
    ok(float(np.sqrt(np.mean(lead ** 2))) < 0.01, "the clip starts before the first loud sound (pre-roll)")
    ok(not det.active, "detector is idle again")

    # ---- a short pause inside a sentence does not split it
    out, _ = run(np.concatenate([room(0.6), speech(0.8), room(0.5), speech(0.8), room(1.4)]))
    ok(len(out) == 1 and out[0][1] > 1.4, "a 0.5 s pause stays inside one utterance")

    # ---- two sentences with a long pause = two utterances
    out, _ = run(np.concatenate([room(0.6), speech(0.8), room(1.5), speech(0.8), room(1.4)]))
    ok(len(out) == 2, "a 1.5 s pause splits two utterances")

    # ---- a blip is reported as too short (the caller drops it); silence gives nothing
    out, _ = run(np.concatenate([room(0.6), speech(0.2), room(1.4)]))
    ok(all(s < app.VAD_MIN_SPEECH_S for _, s in out), "a 0.2 s blip is under the minimum")
    ok(run(room(5.0))[0] == [], "a quiet room gives nothing")
    ok(run(np.zeros(app.RATE * 3, np.float32))[0] == [], "digital silence gives nothing")

    # ---- nonstop talking is cut at the maximum length
    out, _ = run(np.concatenate([room(0.6), speech(20.0), room(1.4)]))
    ok(out and abs(out[0][0].size / app.RATE - app.VAD_MAX_S) < 0.1, "cut at the maximum length")

    # ---- a steady noise (fan, music) stops counting as speech
    out, det = run(np.concatenate([room(1.0), speech(90.0, level=0.02)]))
    ok(len(out) == 1 and not det.active, f"steady noise: one clip, then ignored (got {len(out)})")
    out2 = []
    louder = np.concatenate([speech(1.0, level=0.02), speech(1.2, level=0.2), speech(1.5, level=0.02)])
    for i in range(0, louder.size, 512):
        out2 += det.feed(louder[i:i + 512])
    ok(len(out2) == 1 and out2[0][1] >= 1.0, "a voice well above that noise is still heard")

    # ---- a 48 kHz microphone, odd block sizes
    out, _ = run(np.concatenate([room(1.0, 48000), speech(1.0, 48000), room(1.4, 48000)]), rate=48000, block=441)
    ok(len(out) == 1 and 0.8 <= out[0][1] <= 1.2, "48 kHz microphone")
    ok(abs(app.to_16k(out[0][0], 48000).size / app.RATE - out[0][0].size / 48000) < 0.01, "resampled to 16 kHz")

    # ---- --wav test microphone: one LISTEN_ON hears the file once
    path = os.path.join(tmp, "hello.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(app.RATE)
        w.writeframes((np.concatenate([room(0.6), speech(1.0), room(0.3)]) * 32767).astype(np.int16).tobytes())
    rec = app.WavRecorder(path)
    tap, det, out = rec.open_tap(), app.UtteranceDetector(rec.rate), []
    while not tap.empty():
        out += det.feed(tap.get())
    ok(len(out) == 1 and out[0][1] >= 0.8, "WavRecorder feeds gaze mode")

    # ---- mode from .env
    for given, want in (("", "gaze"), ("ptt", "ptt"), (" PTT ", "ptt"), ("gaze", "gaze"), ("always", "gaze")):
        os.environ["W04_LISTEN_MODE"] = given
        ok(app.listen_mode_from_env() == want, f"W04_LISTEN_MODE={given!r} -> {want}")
    os.environ.pop("W04_LISTEN_MODE")
    ok(app.listen_mode_from_env() == "gaze", "no W04_LISTEN_MODE -> gaze")


def start_watcher(tmp, name, mode, make_brain=SlowBrain):
    persona = os.path.join(tmp, name + "_persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    store = persona_store.Store(data_dir=os.path.join(tmp, name + "_data"), persona_path=persona)
    bridge = cfg_bridge.CfgBridge(store, set(), log=app.log)
    brain, tr, mic = make_brain(store), FakeTr(), FakeMic()
    logs = os.path.join(tmp, name + "_vrc")
    os.makedirs(logs)
    path = os.path.join(logs, "output_log_2026-11-19_13-44-33.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(join(LOCAL) + T + '[Behaviour] Initialized PlayerAPI "Tester One" is local\n' + LISTEN_ON)
    while not app.replies.empty():
        app.replies.get_nowait()
    stop = threading.Event()
    watcher = threading.Thread(target=app.watch_log, args=(logs, mic, tr, False, brain, store, bridge, stop),
                               kwargs={"listen_mode": mode}, daemon=True)
    watcher.start()
    wait_for(lambda: store.speaker == "Tester One", "watcher read the backlog")
    return path, store, bridge, brain, tr, mic, stop, watcher


def add(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def tsv(kind):
    with open(app.LOG_PATH, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t", 2)[2] for line in f if line.split("\t")[1] == kind]


def no_reply(seconds=0.4):
    time.sleep(seconds)
    return app.replies.empty()


def gaze_checks(tmp):
    path, store, bridge, brain, tr, mic, stop, watcher = start_watcher(tmp, "gaze", "gaze")
    try:
        # ---- not listening: the microphone is not even read
        time.sleep(0.3)
        ok(mic.tap is None and not tsv("LISTEN_ON"), "a LISTEN_ON from before the app started is not replayed")
        mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
        ok(mic.tap is None and mic.unheard > 0 and no_reply() and not brain.heard and not tr.clips,
           "speech while LISTEN is off goes nowhere")

        # ---- LISTEN_ON, speech, silence -> one utterance -> the AI answers
        add(path, LISTEN_ON)
        wait_for(lambda: mic.tap is not None, "microphone opened by LISTEN_ON")
        tr.text = "창밖 야경 예쁘다"
        mic.push(np.concatenate([room(0.6), speech(1.2), room(1.4)]))
        reply = app.replies.get(timeout=5)
        ok(reply["line"] == "응! 창밖 야경 예쁘다" and reply["heard"] == "창밖 야경 예쁘다" and brain.heard == ["창밖 야경 예쁘다"],
           "speech then silence: one utterance, answered")
        ok(len(tr.clips) == 1 and 1.2 < tr.clips[0] < 2.2, f"Whisper got one clip of the right length ({tr.clips})")
        ok(len(tsv("VAD_END")) == 1 and len(tsv("LISTEN_ON")) == 1, "LISTEN_ON and VAD_END are logged")
        ok(no_reply() and mic.tap is not None, "still listening, nothing more said")

        # ---- a blip is ignored
        mic.push(np.concatenate([speech(0.2), room(1.4)]))
        wait_for(lambda: tsv("VAD_BLIP"), "blip seen")
        ok(no_reply() and len(tr.clips) == 1 and len(brain.heard) == 1, "a blip is not sent to Whisper or the AI")

        # ---- a sound Whisper finds no words in: no "응? 잘 못 들었어." in open-mic mode
        tr.text = ""
        mic.push(np.concatenate([speech(1.0), room(1.4)]))
        wait_for(lambda: len(tr.clips) == 2, "noise reached Whisper")
        ok(no_reply() and len(brain.heard) == 1, "an utterance without words gets no reply")

        # ---- LISTEN_OFF in the middle of a sentence: the sentence is finished
        tr.text = "끝까지 들어 줘"
        wait_for(lambda: len(tsv("STT")) == 2, "previous turn done")
        mic.push(np.concatenate([room(0.3), speech(1.0)]))
        wait_for(mic.drained, "first half heard")
        time.sleep(0.2)
        add(path, LISTEN_OFF)
        wait_for(lambda: len(tsv("LISTEN_OFF")) == 1, "LISTEN_OFF seen")
        time.sleep(0.2)
        ok(mic.tap is not None, "the microphone stays open while the utterance is in progress")
        mic.push(np.concatenate([speech(0.6), room(1.4)]))
        reply = app.replies.get(timeout=5)
        ok(reply["heard"] == "끝까지 들어 줘" and 1.6 < tr.clips[-1] < 2.6,
           f"the utterance cut by LISTEN_OFF is whole ({tr.clips[-1]:.2f} s)")
        wait_for(lambda: mic.tap is None, "microphone closed after the utterance")
        before = len(tr.clips)
        mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
        ok(no_reply() and len(tr.clips) == before, "after LISTEN_OFF nothing more is heard")

        # ---- one turn at a time: speech during an answer is dropped, the next one is heard
        add(path, LISTEN_ON)
        wait_for(lambda: mic.tap is not None, "listening again")
        brain.gate.clear()
        tr.text = "첫 번째"
        mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
        wait_for(lambda: brain.heard[-1] == "첫 번째", "AI is thinking about the first")
        tr.text = "두 번째"
        mic.push(np.concatenate([speech(1.0), room(1.4)]))
        wait_for(mic.drained, "second heard by the microphone")
        time.sleep(0.2)
        brain.gate.set()
        reply = app.replies.get(timeout=5)
        ok(reply["heard"] == "첫 번째" and no_reply() and "두 번째" not in brain.heard,
           "speech while an answer is being made is dropped, not queued")
        tr.text = "세 번째"
        mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
        ok(app.replies.get(timeout=5)["heard"] == "세 번째", "the next utterance after the answer is heard")

        # ---- push-to-talk still works in gaze mode (and its missed clip still answers)
        tr.text = "점프로 말함"
        add(path, PTT)
        ok(app.replies.get(timeout=5)["heard"] == "점프로 말함", "push-to-talk works in gaze mode")
        tr.text = ""
        add(path, PTT)
        ok(app.replies.get(timeout=5)["line"] == app.MISSED_LINE, "a missed push-to-talk clip still says so")

        # ---- dictation armed from the settings panel: the next spoken utterance goes there, no AI
        add(path, LISTEN_OFF)
        wait_for(lambda: mic.tap is None, "closed")
        add(path, T + '[W04AI] CFG_OPEN utc=1\n' + T + '[W04AI] CFG_SET {"op":"dictate","target":"note_add"} utc=1\n')
        wait_for(lambda: mic.tap is not None, "an armed dictation opens the microphone without LISTEN_ON")
        tr.text = "루나는 딸기를 좋아한다"
        heard = len(brain.heard)
        mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
        wait_for(lambda: store.notes and store.notes[-1]["text"] == "루나는 딸기를 좋아한다", "dictated by voice")
        ok(len(brain.heard) == heard and no_reply(), "the dictated utterance did not go to the AI")
        wait_for(lambda: mic.tap is None, "microphone closed after the dictation")

        # ---- the VRChat log stops (crash): listening ends by itself
        add(path, LISTEN_ON)
        wait_for(lambda: mic.tap is not None, "listening")
        app.LISTEN_STALE_S = 0.3
        wait_for(lambda: any("stale_log" in d for d in tsv("LISTEN_OFF")), "stale log ends listening")
        wait_for(lambda: mic.tap is None, "microphone closed")
        app.LISTEN_STALE_S = 120

        # ---- leaving the world ends listening; a world that is not allowed cannot start it
        add(path, LISTEN_ON)
        wait_for(lambda: mic.tap is not None, "listening")
        add(path, T + "[Behaviour] OnLeftRoom\n" + join(OTHER))
        wait_for(lambda: any("world_changed" in d for d in tsv("LISTEN_OFF")), "world change ends listening")
        wait_for(lambda: mic.tap is None, "microphone closed")
        add(path, LISTEN_ON)
        wait_for(lambda: tsv("LISTEN_IGNORED"), "foreign LISTEN_ON seen")
        time.sleep(0.2)
        ok(mic.tap is None, "LISTEN_ON from a world that is not allowed does not open the microphone")
    finally:
        app.LISTEN_STALE_S = 120
        brain.gate.set()
        stop.set()
        watcher.join(timeout=3)


def ptt_checks(tmp):
    path, store, bridge, brain, tr, mic, stop, watcher = start_watcher(tmp, "ptt", "ptt")
    try:
        ignored = len(tsv("LISTEN_IGNORED"))
        add(path, LISTEN_ON)
        wait_for(lambda: len(tsv("LISTEN_IGNORED")) == ignored + 1, "LISTEN_ON seen in ptt mode")
        mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
        ok(mic.tap is None and no_reply() and not brain.heard, "ptt mode: LISTEN_ON does not open the microphone")
        tr.text = "안녕 반가워"
        add(path, PTT)
        reply = app.replies.get(timeout=5)
        ok(reply["line"] == "응! 안녕 반가워" and brain.heard == ["안녕 반가워"], "ptt mode: push-to-talk answers as before")
        tr.text = ""
        add(path, PTT)
        ok(app.replies.get(timeout=5)["line"] == app.MISSED_LINE, "ptt mode: a missed clip still says so")
        add(path, T + "[W04AI] PTT_DOWN utc=1\n" + T + "[W04AI] PTT_UP held_ms=120 utc=2\n")
        ok(app.replies.get(timeout=5)["line"] == app.MISSED_LINE and len(tr.clips) == 2, "ptt mode: a tap is too short")
    finally:
        stop.set()
        watcher.join(timeout=3)


def overheard_checks(tmp):
    fake_model_calls = []

    def make_brain(store):
        brain = app.persona_ai.Brain("fake", store)
        brain.client = object()

        def ask(request, _usage):
            check = "for_me" in request["output_config"]["format"]["schema"]["required"]
            fake_model_calls.append(check)
            if check and "질문 좀 해도 돼?" in request["messages"][-1]["content"]:
                return {"for_me": True, "lines": {"ko": "응!"}}
            # Even if a skipped response also includes a search, a line and a memory, none are used.
            if check:
                return {"for_me": False, "lines": {"ko": "응!"}, "steps": [], "search": "날씨",
                        "memory_note": "저장하면 안 됨"}
            return {"lines": {"ko": "응!"}}

        brain._ask = ask
        return brain

    path, store, _, _, tr, mic, stop, watcher = start_watcher(tmp, "overheard", "gaze", make_brain)
    try:
        with patch.dict(os.environ, {"W04_WEB_SEARCH": "1"}):
            add(path, LISTEN_ON)
            wait_for(lambda: mic.tap is not None, "address filtering listener is active")
            tr.text = "민수야 오늘 날씨 어때?"
            mic.push(np.concatenate([room(0.6), speech(1.0), room(1.4)]))
            wait_for(lambda: tsv("AI_SKIP"), "overheard speech logged")
            ok(app.replies.empty() and fake_model_calls == [True], "gaze skip is silent and uses one call")
            ok(store.history[0]["content"] == "(다른 사람에게 한 말로 판단)" and not store.notes
               and not store.players, "skip keeps only a marker without touching impression or notes")
            add(path, PTT)
            skipped = app.replies.get(timeout=5)
            ok(skipped["line"] == "" and skipped["steps"] == [], "PTT skip closes the poll without speech")
            tr.text = "권 준아 오늘 날씨 좋다"
            add(path, PTT)
            ok(app.replies.get(timeout=5)["line"] == "응!", "spoken NPC name bypasses address judgment")
            tr.text = "그래서 산책 갈까?"
            add(path, PTT)
            ok(app.replies.get(timeout=5)["line"] == "응!", "follow-up bypasses address judgment")
            with patch.object(app, "FOLLOW_UP_MS", 0):
                add(path, PTT)
                ok(app.replies.get(timeout=5)["line"] == "", "address judgment resumes after the window")
                tr.text = "질문 좀 해도 돼?"
                add(path, PTT)
                ok(app.replies.get(timeout=5)["line"] == "응!", "a positive address judgment answers normally")
                store.apply({"settings": {"filter_overheard": False}})
                add(path, PTT)
                ok(app.replies.get(timeout=5)["line"] == "응!", "disabled setting bypasses address judgment")
            ok(fake_model_calls == [True, True, False, False, True, True, False], "one model call for each utterance")
    finally:
        stop.set()
        watcher.join(timeout=3)


def main():
    tmp = tempfile.mkdtemp(prefix="w04_gaze_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")        # keep the real log clean
        detector_checks(tmp)
        gaze_checks(tmp)
        ptt_checks(tmp)
        overheard_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_regression_checks():
    main()


if __name__ == "__main__":
    main()
