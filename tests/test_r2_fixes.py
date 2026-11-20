"""Offline second-round checks. Run directly with .venv\Scripts\python.exe -X utf8 tests\test_r2_fixes.py."""
import contextlib
import io
import os
import queue
import tempfile
import threading
from unittest.mock import patch

import numpy as np

import test_gaze_listen as gaze
from test_review_fixes import down, start, up

import persona_ai
import persona_npc_app as app

checks = 0


def ok(condition, what):
    global checks
    if not condition:
        raise AssertionError(what)
    checks += 1


def vad_checks():
    with patch.dict(os.environ):
        os.environ.pop("W04_END_SILENCE_S", None)
        det = app.UtteranceDetector(app.RATE)
        ok(det.end_silence == 1.3, "default silence is 1.3 seconds")
        ok(not det.feed(np.concatenate([gaze.speech(0.8), gaze.room(1.0)])), "one second does not split")
        out = det.feed(np.concatenate([gaze.speech(0.8), gaze.room(1.4)]))
        ok(len(out) == 1 and out[0][1] > 1.4, "both halves are in the same recording")
        os.environ["W04_END_SILENCE_S"] = "0.6"
        det = app.UtteranceDetector(app.RATE)
        ok(len(det.feed(np.concatenate([gaze.speech(0.8), gaze.room(0.8)]))) == 1, "honors override")
        for invalid in ("bad", "nan", "inf", "0", "-1", "15"):
            os.environ["W04_END_SILENCE_S"] = invalid
            ok(app.end_silence_from_env() == 1.3, f"invalid silence {invalid} falls back")
    ok(app.is_fragment("응!") and app.is_fragment("시식을") and app.is_fragment("창가에 앉아서"),
       "short cries and unfinished speech can be continued")
    ok(not app.is_fragment("창가에 앉았어요") and not app.is_fragment("잊을 사람들?"), "sentence endings close a turn")


def push(mic):
    mic.push(np.concatenate([gaze.speech(0.9), gaze.room(1.4)]))


def gaze_checks(tmp):
    path, store, _, brain, tr, mic, stop, watcher = gaze.start_watcher(tmp, "r2_gaze", "gaze")
    try:
        gaze.add(path, gaze.T + "[W04AI] NPC_POSE npc=kwonjun pose=sit\n" + gaze.LISTEN_ON)
        gaze.wait_for(lambda: mic.tap is not None and store.pose == "sit", "listening while seated")
        inputs = iter(("시식을", "잊을 사람들?", "다음 이야기는", "흐" * 80 + " 정말 웃기네요!"))
        original_run = tr.run

        def run(audio):
            tr.text = next(inputs)
            return original_run(audio)

        tr.run = run
        push(mic)
        gaze.wait_for(lambda: len(tr.clips) == 1, "first clip transcribed")
        ok(not brain.heard, "fragment waits for a continuation")
        push(mic)
        reply = app.replies.get(timeout=5)
        ok(reply["heard"] == "시식을 잊을 사람들?" and brain.heard == [reply["heard"]], "combined before AI")
        ok(reply["steps"] == [], "conversation while seated does not inject a new sit step")
        gaze.wait_for(lambda: len(gaze.tsv("REPLY")) == 1, "worker finished the merged turn")
        push(mic)
        ok(app.replies.get(timeout=5)["heard"] == "다음 이야기는", "uncontinued fragment is delivered after the window")
        gaze.wait_for(lambda: len(gaze.tsv("REPLY")) == 2, "worker finished the standalone fragment")
        push(mic)
        reply = app.replies.get(timeout=5)
        ok(brain.heard[-1] == reply["heard"] == "흐흐흐 정말 웃기네요!", "AI and world see collapsed laughter")
        ok(any(reply["heard"] in line for line in gaze.tsv("STT"))
           and all("흐" * 6 not in line for line in gaze.tsv("STT")), "STT log also sees shortened transcript")
        ok(app.collapse_transcript_runs("하하 " * 20 + "123333333") == "하하하 123333333"
           and app.collapse_transcript_runs("으흐" * 80) == "으흐으흐", "spaced / alternating syllables collapse; digits stay")
        gaze.wait_for(lambda: len(gaze.tsv("REPLY")) == 3, "worker finished the laughter turn")
        tr.run = original_run
        original_reply = brain.reply
        for pose, text, action in (("sit", "창가로 가", {"do": "move_to", "target": "window"}),
                                   ("lie", "따라와", {"do": "follow"}),
                                   ("lie", "화면 130인치로 해 줘", {"do": "tv_size_set", "inches": 130})):
            gaze.add(path, gaze.T + f"[W04AI] NPC_POSE npc=kwonjun pose={pose}\n")
            gaze.wait_for(lambda: store.pose == pose, "watcher saw the current pose")
            tr.text = text
            with patch.object(brain, "reply", side_effect=lambda heard, **kw:
                              dict(original_reply(heard, **kw), steps=persona_ai.clamp_steps([action]))):
                push(mic)
                reply = app.replies.get(timeout=5)
            ok(reply["steps"] == persona_ai.clamp_steps([action]), f"{pose}: world receives the command {action['do']}")
    finally:
        stop.set()
        watcher.join(timeout=3)


class SearchBrain:
    def __init__(self, search_text="clip 1"):
        self.search_text = search_text
        self.release = threading.Event()
        self.finished = threading.Event()

    def reply(self, text, speaker=None, on_interim=None):
        def payload(line):
            return {"emotion": "neutral", "intensity": 0.5, "steps": [], "line": line, "lines": {"ko": line},
                    "model": "fake", "ai_ms": 1}

        if text == self.search_text:
            on_interim(dict(payload("잠깐만, 찾아볼게!"), interim="search"))
            self.release.wait(5)
            self.finished.set()
            return payload("찾아보니 오늘은 맑대!")
        return dict(payload("다음 말도 들었어!"), remembered={"memory_note": "귤 좋아함."})


def search_checks(tmp):
    brain = SearchBrain()
    path, _, rec, _, stop, watcher = start(tmp, "r2_search", brain)
    try:
        add = gaze.add
        add(path, down("kwonjun") + up("kwonjun"))
        interim = app.replies.get_for_world(5)
        ok(interim["more"] and interim["line"] == "잠깐만, 찾아볼게!", "world receives the search notice")
        add(path, down("kwonjun") + up("kwonjun"))
        gaze.wait_for(lambda: rec.stops == 2, "second utterance recorded during search")
        brain.release.set()
        gaze.wait_for(lambda: app.replies.qsize() == 2, "both final replies queued")
        app.clear_replies(preserve_search=True)
        final = app.replies.get_for_world(5)
        ok(final["line"] == "찾아보니 오늘은 맑대!" and final["seq"] == interim["seq"] and final["more"],
           "the same search final reaches the world after a newer utterance and queue clear")
        app.replies.retry_search(final)
        ok(app.replies.get_for_world(0) is final, "a failed HTTP send can retry the search final immediately")
        app.replies.put({"npc": "kwonjun", "line": "다음 대답"})
        app.replies.put({"npc": "itoa", "line": "다른 NPC"})
        ok(app.replies.get_for_world(0)["npc"] == "itoa", "other NPC can deliver its reply immediately")
        try:
            app.replies.get_for_world(0)
        except queue.Empty:
            ok(True, "same-NPC follow-up cannot replace the pending search answer")
        else:
            raise AssertionError("same-NPC follow-up displaced the search result")
    finally:
        brain.release.set()
        stop.set()
        watcher.join(timeout=3)


def gaze_search_checks(tmp):
    search = SearchBrain("날씨 검색해 줘!")
    path, _, _, brain, tr, mic, stop, watcher = gaze.start_watcher(tmp, "r2_gaze_search", "gaze")
    try:
        brain.reply = search.reply
        gaze.add(path, gaze.LISTEN_ON)
        gaze.wait_for(lambda: mic.tap is not None, "gaze microphone ready")
        tr.text = search.search_text
        push(mic)
        ok(app.replies.get(timeout=5)["more"], "gaze search started")
        tr.text = "이 말도 받아 줘!"
        push(mic)
        gaze.wait_for(mic.drained, "later gaze audio consumed during search")
        search.release.set()
        final, later = app.replies.get(timeout=5), app.replies.get(timeout=5)
        ok(final["line"] == "찾아보니 오늘은 맑대!" and later["heard"] == tr.text,
           "gaze speech during search is processed after the final answer")
    finally:
        search.release.set()
        stop.set()
        watcher.join(timeout=3)


def switched_off_checks(tmp):
    for queued in (False, True):
        with patch.object(app, "replies", app.ReplyQueue()):
            brain = SearchBrain()
            path, store, _, _, stop, watcher = start(tmp, f"r2_off_{queued}", brain)
            try:
                gaze.add(path, down("kwonjun") + up("kwonjun"))
                ok(app.replies.get(timeout=5)["more"], "interim before switching off")
                if queued:
                    brain.release.set()
                    gaze.wait_for(lambda: app.replies.qsize() == 1, "final queued before switching off")
                gaze.add(path, gaze.T + "[W04AI] NPC_STATE kwonjun=0 itoa=0\n"
                         + gaze.T + "[W04AI] NPC_STATE kwonjun=1 itoa=0\n"
                         + gaze.T + "[W04AI] NPC_POSE npc=kwonjun pose=sit\n")
                gaze.wait_for(lambda: store.pose == "sit", "off and on processed before the pose line")
                brain.release.set()
                gaze.wait_for(brain.finished.is_set, "search call finished")
                ok(gaze.no_reply(0.2), "turning off drops the old search even if the NPC is reenabled")
            finally:
                brain.release.set()
                stop.set()
                watcher.join(timeout=3)


def main():
    with tempfile.TemporaryDirectory(prefix="w04_r2_test_") as tmp:
        env = {"W04_END_SILENCE_S": "1.3"}
        with patch.object(app, "LOG_PATH", os.path.join(tmp, "voice_log.tsv")), patch.dict(os.environ, env):
            vad_checks()
            gaze_checks(tmp)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), patch.object(app, "replies", app.ReplyQueue()):
                search_checks(tmp)
            ok("기억에 기록함: 귤 좋아함." in out.getvalue() and "둠둠" not in out.getvalue(), "console memory label")
            with patch.object(app, "replies", app.ReplyQueue()):
                gaze_search_checks(tmp)
            switched_off_checks(tmp)
    ok("비밀" in persona_ai.RULES and "어떤 경우에도" in persona_ai.RULES, "persona has model privacy rule")
    for line in ("나는 Claude Haiku 5.5야.", "앤트로픽이 만들었어.", "claude-sonnet-5"):
        reply = persona_ai._clamp_reply({"lines": {"ko": line}})
        ok(reply["line"] == reply["lines"]["ko"] == persona_ai.MODEL_SECRET_LINE, "model leakage is replaced")
    print(f"OK: {checks} checks passed")


if __name__ == "__main__":
    main()
