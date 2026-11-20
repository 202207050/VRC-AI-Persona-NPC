"""Offline VAD and Whisper regression checks. No microphone or GPU needed."""
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import persona_npc_app as app
from voice_options import whisper_beam_size


def test_adaptive_vad():
    det = app.UtteranceDetector(app.RATE)

    def feed(rms, frames):
        # Match the frame RMS exactly, including at the start and end of short sounds.
        wave = np.tile(np.array([rms, -rms], np.float32), det.n * frames // 2)
        return det.feed(wave)

    assert not feed(0.0004, 40)
    assert det.onset_threshold == app.VAD_MIN_THRESHOLD
    assert not feed(0.004, 7)
    [(clip, voiced)] = feed(0.0004, 50)
    assert voiced == app.VAD_MIN_SPEECH_S
    assert app.VAD_MIN_CLIP_RMS < np.sqrt(np.mean(clip ** 2)) < app.MIN_RMS

    for frames in (1, 6):
        out = feed(0.05, frames) + feed(0.0004, 50)
        assert all(speech < app.VAD_MIN_SPEECH_S for _, speech in out)

    det = app.UtteranceDetector(app.RATE)
    assert not feed(0.002, 90)
    assert det.onset_threshold > 0.004
    assert not feed(0.004, 10) and not feed(0.002, 50)
    assert not feed(0.035, 20)
    [(clip, voiced)] = feed(0.002, 50)
    assert voiced >= 0.6


def result(text, no_speech=0.15, logprob=-0.3):
    return {"text": text, "segments": [{"no_speech_prob": no_speech, "avg_logprob": logprob}]}


def test_gates_and_retry():
    tr = app.Transcriber.__new__(app.Transcriber)
    tr.model, tr.device = Mock(), "cpu"
    audio = np.full(app.RATE, 0.002, np.float32)
    voiced = app.VAD_MIN_SPEECH_S

    tr.model.transcribe.return_value = result("앉아")
    assert app.judge(audio, -1, tr, vad_speech_s=voiced)[0] == "앉아"
    assert app.judge(audio, 600, tr)[1].startswith("too_quiet")
    assert app.judge(audio * 20, 150, tr)[1].startswith("too_short")

    for no_speech, logprob in ((0.6, -0.3), (0.15, -0.8)):
        tr.model.transcribe.return_value = result("거울", no_speech, logprob)
        tr.model.transcribe.reset_mock()
        assert app.judge(audio, -1, tr, vad_speech_s=voiced)[:2] == ("", "weak_speech")
        assert tr.model.transcribe.call_count == 1

    for filler in ("어?", "아아아!", "헐", "ㄱ", "마대"):
        tr.model.transcribe.return_value = result(filler)
        tr.model.transcribe.reset_mock()
        assert app.judge(audio, -1, tr, vad_speech_s=voiced)[:2] == ("", "unclear_short")
        first, retry = tr.model.transcribe.call_args_list
        assert "권준" in first.kwargs["initial_prompt"]
        assert retry.kwargs["initial_prompt"] is None

    tr.model.transcribe.side_effect = [result("아"), result("TV 켜")]
    raw = []
    assert app.judge(audio, -1, tr, raw_sink=raw, vad_speech_s=voiced)[0] == "TV 켜"
    assert raw == ["TV 켜"]
    tr.model.transcribe.side_effect = [result("어?"), result("따라와", no_speech=0.55)]
    assert app.judge(audio, -1, tr, vad_speech_s=voiced)[:2] == ("", "unclear_short")
    tr.model.transcribe.side_effect = None

    for short in ("앉아", "켜", "응", "네", "권준", "거울", "7", "별"):
        tr.model.transcribe.return_value = result(short)
        tr.model.transcribe.reset_mock()
        assert app.judge(audio, -1, tr, hints="별", vad_speech_s=voiced)[0] == short
        assert tr.model.transcribe.call_count == 1


def test_beam_option():
    tr = app.Transcriber.__new__(app.Transcriber)
    tr.model, tr.device = Mock(), "cuda"
    audio = np.zeros(app.RATE, np.float32)
    for setting in ("", "1", "invalid", "2"):
        tr.beam_size = whisper_beam_size(setting)
        tr.run(audio)
        assert "beam_size" not in tr.model.transcribe.call_args.kwargs
    tr.beam_size = whisper_beam_size(" 5 ")
    tr.run(audio)
    assert tr.model.transcribe.call_args.kwargs["beam_size"] == 5
    assert tr.model.transcribe.call_args.kwargs["temperature"] == 0.0


if __name__ == "__main__":
    test_adaptive_vad()
    test_gates_and_retry()
    test_beam_option()
    print("adaptive STT checks passed")
