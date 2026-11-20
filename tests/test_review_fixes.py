"""Offline checks for the 2026-11-19 review fixes and the new step types.

    .venv\\Scripts\\python.exe -X utf8 tests\\test_review_fixes.py

No microphone, no Whisper, no API call, no real VRChat log (fakes and a temporary folder only).
Covers: sit/lie/stand, follow/stop/come/go_sit and tv_* steps, bridge_view (old name "door"), the
NPC_POSE prompt note, AFFECTION -> impression note, push-to-talk recordings detached at release,
push-to-talk cancelled on world change / after the time limit, the worker surviving errors, the
compaction undo file in the data migration, the echo line cut to the world's limit, /cfg default NPC.
"""
import collections
import json
import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.pop("PERSONA_FILE", None)

import cfg_bridge              # noqa: E402
import export_cloud_prompt     # noqa: E402
import persona_ai              # noqa: E402
import persona_npc_app as app  # noqa: E402
import persona_store           # noqa: E402
from test_gaze_listen import LOCAL, OTHER, T, add, join, tsv, wait_for   # noqa: E402
from test_settings import request                                        # noqa: E402

checks = 0
NEW_ACTIONS = ["sit", "lie", "stand", "follow", "stop", "come", "go_sit", "tv_closer", "tv_farther", "tv_lower",
               "tv_raise", "tv_tilt_down", "tv_tilt_up", "tv_left", "tv_right", "tv_reset"]


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def new_store(tmp, name):
    persona = os.path.join(tmp, name + "_persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)
    return persona_store.Store(data_dir=os.path.join(tmp, name + "_data"), persona_path=persona)


# ---------------------------------------------------------------- review #1: new steps

def step_checks(tmp):
    enum = persona_ai.REPLY_SCHEMA["properties"]["steps"]["items"]["properties"]
    for name in NEW_ACTIONS:
        ok(name in persona_ai.ACTIONS and name in enum["do"]["enum"], f"{name} is an action the model may pick")
        out = persona_ai._clamp_reply({"steps": [{"do": name, "target": "sofa", "gesture": "wave"}], "lines": {"ko": "응"}})
        ok(out["steps"] == [{"do": name, "target": "none", "gesture": "none"}], f"{name} survives _clamp_reply with none/none")
    for name in ("look_at", "move_to", "face", "gesture", "wait"):
        ok(name in persona_ai.ACTIONS, f"old action {name} is still there")
    ok("bridge_view" in persona_ai.TARGETS and "door" not in persona_ai.TARGETS
       and "door" not in enum["target"]["enum"], "bridge_view replaced door in the vocabulary")
    out = persona_ai._clamp_reply({"steps": [{"do": "move_to", "target": "door", "gesture": "none"},
                                             {"do": "move_to", "target": "bridge_view", "gesture": "none"},
                                             {"do": "look_at", "target": "nowhere", "gesture": "none"}], "lines": {"ko": "응"}})
    ok([s["target"] for s in out["steps"]] == ["bridge_view", "bridge_view", "speaker"],
       "target door -> bridge_view; an unknown target still falls back to speaker")
    out = persona_ai._clamp_reply({"steps": [{"do": "tv_left", "target": "tv", "gesture": "none"},
                                             {"do": "tv_raise", "target": "none", "gesture": "none"},
                                             {"do": "look_at", "target": "speaker", "gesture": "none"}], "lines": {"ko": "응"}})
    ok([s["do"] for s in out["steps"]] == ["look_at", "tv_left"], "one tv step per reply, moved to the end")
    for name in ("tv_closer_max", "tv_farther_max", "tv_stop", "tv_pause",
                 "tv_size_up", "tv_size_down", "tv_size_max", "tv_size_min", "tv_restart", "tv_resume", "tv_off"):
        ok(name in enum["do"]["enum"] and persona_ai.clamp_steps([{"do": name}]) ==
           [{"do": name, "target": "none", "gesture": "none"}], f"{name} reaches the world")
    for data, field, expected in (
        ({"do": "tv_seek"}, "seconds", 10),
        ({"do": "tv_seek", "seconds": -45}, "seconds", -45),
        ({"do": "tv_backlight", "percent": 0}, "percent", 0),
        ({"do": "tv_backlight", "action": "dimmer"}, "action", "dimmer"),
    ):
        ok(persona_ai.clamp_steps([data])[0][field] == expected, f"{data} forwards {field}")
    ok(not persona_ai.clamp_steps([{"do": "tv_backlight", "percent": True},
                                  {"do": "tv_seek", "seconds": float("inf")}]),
       "foreign TV argument types do not reach the world")
    faces = [{"do": "emote", "emotion": emotion, "duration": duration}
             for emotion, duration in [("happy", 3), ("surprised", 40), ("sad", 0.1)]]
    steps = persona_ai.clamp_steps(faces)
    ok([(s["emotion"], s["duration"]) for s in steps] ==
       [("happy", 3), ("surprised", 10), ("sad", 0.5)], "emotes retain order and bounded durations")
    ok(persona_ai.clamp_steps([{"do": "emote", "emotion": "happy"}])[0]["duration"] == 3
       and not persona_ai.clamp_steps([{"do": "emote", "emotion": "unknown"}]),
       "missing emote duration defaults to 3 and unknown emotions are dropped")
    ok("tv_size_set" in enum["do"]["enum"] and enum["inches"]["type"] == "number"
       and persona_ai.clamp_steps([{"do": "tv_size_set", "inches": 183.5}]) ==
       [{"do": "tv_size_set", "target": "none", "gesture": "none", "inches": 183.5}],
       "exact screen size is passed to the stick for snapping and fitting")
    for value in (None, "180", True, -10, float("nan"), float("inf")):
        ok(persona_ai.clamp_steps([{"do": "tv_size_set", "inches": value}, {"do": "tv_size_min"}]) ==
           [{"do": "tv_size_min", "target": "none", "gesture": "none"}], "invalid size does not shadow a valid TV step")
    ok(persona_ai._clamp_reply({"steps": [None, 3, {"do": "fly"}], "lines": {"ko": "응"}})["steps"] == [],
       "junk and unknown steps are dropped")
    rules = persona_ai.RULES
    for piece in ("- follow:", "시키지 않았는데 따라가지 않는다", "- come:", "- stop:", "- go_sit:", "tv_closer/tv_farther",
                  "TV 명령은 대답당 하나를 마지막에 넣고", 'target과 gesture를 "none"으로 둔다', "sit: 소파에 가서 앉는다",
                  "대화 중에는 현재 자세를 유지한다", '"계속해"에는 새 이동·자세·TV 동작을 넣거나',
                  'target "bridge_view"는 다리가 보이는 창가다'):
        ok(piece in rules, f"RULES has: {piece}")

    # current pose (NPC_POSE) goes into the changing part of the prompt, never the cached part
    store = new_store(tmp, "steps")
    brain = persona_ai.Brain("fake", store)
    fixed0, changing0 = brain.system_parts()
    metadata = {"speaker": "원문 시험", "source": "PC 주인 마이크", "time": "2026-11-20T12:00:00+09:00",
                "raw": "  사공사  흐흐흐흐흐? "}
    store.after_reply("원문 시험", "404 흐흐흐?", '{"lines":{"ko":"응"}}', utterance=metadata)
    quoted = persona_ai._clamp_reply({"lines": {"ko": "방금 “12345개”라고 했어."}}, prior_raw=" 12345개 ")
    ok(quoted["line"] == "방금 “ 12345개 ”라고 했어.", "quoted STT keeps numbers and spacing verbatim")
    persisted = store._clean_history(json.loads(json.dumps(store.history, ensure_ascii=False)))
    ok(persisted[-2]["utterance"] == metadata, "history retains exact STT and attribution after reloading")
    for level in (0, 1, 2):
        request = brain._request_kwargs("방금 뭐라고 했어?", "원문 시험", content_level=level, utterance=metadata)
        ok(request["system"][0]["text"] == fixed0 and any(
            block["text"] == persona_ai.content_order(level) for block in request["system"][1:]),
           f"conversation level {level} follows the fixed cache block")
        ok(metadata["raw"] in request["messages"][-3]["content"],
           f"conversation level {level} retains the previous raw transcription in history")
    ok("지금 내 자세" not in changing0, "standing: no pose note")
    store.pose = "sit"
    fixed1, changing1 = brain.system_parts()
    ok(fixed1 == fixed0 and "소파에 앉아 있다" in changing1 and "월드가 먼저 일으켜" in changing1,
       "sitting is told in the changing part and the world handles standing before movement")
    store.pose = "lie"
    ok("소파에 누워 있다" in brain.system_parts()[1], "lying is told too")
    ok("따라다니기" in brain.about_me() and "화면 최대 크기" in brain.about_me() and "문 쪽" not in brain.about_me(),
       "about me lists the new abilities")

    # cloud: no follow / come, and the exported file on disk is current
    data = export_cloud_prompt.build_all()
    ok("follow" not in data["actions"] and "come" not in data["actions"]
       and data["schema"]["properties"]["steps"]["items"]["properties"]["do"]["enum"] == data["actions"],
       "cloud actions leave out follow and come")
    ok(all(a in data["actions"] for a in NEW_ACTIONS if a not in ("follow", "come")), "cloud keeps sit/lie/stand/stop/go_sit/tv_*")
    ok("bridge_view" in data["targets"] and "door" not in data["targets"], "cloud targets: bridge_view")
    for npc in persona_store.NPC_IDS:
        text = data["personas"][npc]["system_fixed"]
        ok("- follow:" not in text and "- come:" not in text and "- go_sit:" in text and "- stop:" in text,
           f"cloud rules for {npc} have no follow / come line")
    ok("- follow:" in persona_ai.RULES, "the app's own rules were not changed by the export")
    with open(export_cloud_prompt.OUT_PATH, encoding="utf-8") as f:
        on_disk = json.load(f)
    if not os.environ.get("PERSONA_FILE"):
        ok(on_disk == data, "Tools/cloud/src/prompt.json is up to date (run export_cloud_prompt.py)")


# ---------------------------------------------------------------- review #9: AFFECTION

def affection_store_checks(tmp):
    store = new_store(tmp, "aff")
    itoa = store.for_npc("itoa")
    pat = persona_store.AFFECTION_PHRASES["pat"]
    hug = persona_store.AFFECTION_PHRASES["hug"]
    t0 = 1_000_000.0
    ok(itoa.add_affection("Alice", "pat", now=t0) == pat and itoa.players["Alice"]["note"] == pat,
       "a pat is noted in that NPC's impression of that player")
    ok("Alice" not in store.players, "the other NPC's store is untouched")
    ok(itoa.add_affection("Alice", "pat", now=t0 + 599) == "", "same player, same type within 10 minutes: nothing")
    ok(itoa.add_affection("Alice", "hug", now=t0 + 1) == hug and itoa.players["Alice"]["note"] == pat + " " + hug,
       "another type is noted next to it")
    ok(itoa.add_affection("Bob", "pat", now=t0 + 2) == pat, "another player has their own limit")
    ok(itoa.add_affection("Alice", "pat", now=t0 + 601) == "" and itoa.players["Alice"]["note"].count(pat) == 1,
       "after 10 minutes the same phrase is still not repeated")
    itoa.set_impression("Alice", "귤을 좋아함.")
    ok(itoa.add_affection("Alice", "pat", now=t0 + 1300) == "" and itoa.players["Alice"]["note"] == "귤을 좋아함.",
       "the recorded pat stays deduplicated after the impression is edited")
    ok(itoa.add_affection("Alice", "kiss", now=t0) == "" and itoa.add_affection("", "pat", now=t0) == "",
       "unknown type / empty name: nothing")
    itoa.set_locked("Carol", True)
    ok(itoa.add_affection("Carol", "pat", now=t0) == "" and itoa.players["Carol"]["note"] == "", "a locked impression is left alone")
    itoa.apply({"settings": {"auto_update": False}})
    ok(itoa.add_affection("Dave", "pat", now=t0) == "" and "Dave" not in itoa.players, "auto update off: nothing")
    itoa.apply({"settings": {"auto_update": True}})
    itoa.set_impression("Eve", "가" * (persona_store.MAX_IMPRESSION_CHARS - 3))
    ok(itoa.add_affection("Eve", "pat", now=t0) == "", "an impression that is already full is not cut or grown")
    again = persona_store.Store(data_dir=store.root_dir, persona_path=store.persona_path).for_npc("itoa")
    ok(again.players["Bob"]["note"] == pat, "the note is saved to disk")
    full = new_store(tmp, "aff_full")
    for i in range(persona_store.MAX_PLAYERS):
        full.set_impression(f"p{i}", "x")
    ok(full.add_affection("Stranger", "hug", now=t0) == "" and len(full.players) == persona_store.MAX_PLAYERS,
       "a full player list: nobody is pushed out for it")


# ---------------------------------------------------------------- watcher harness

class PressRec:
    """Push-to-talk microphone stand-in: the n-th press records n seconds. No open_tap -> ptt mode."""
    name = "press mic"
    rate = app.RATE

    def __init__(self):
        self.presses = self.stops = self.cancels = 0

    def start(self):
        self.presses += 1

    def stop(self):
        self.stops += 1
        return (np.ones(self.presses * app.RATE) * 0.1).astype(np.float32)

    def cancel(self):
        self.cancels += 1


class ClipTr:
    """Whisper stand-in: a clip of n seconds 'says' "clip n". Can be held back or made to fail."""

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.started = 0
        self.fail = []
        self.text = None

    def run(self, audio):
        self.started += 1
        self.gate.wait(10)
        if self.fail:
            raise self.fail.pop(0)
        return {"text": self.text or f"clip {round(audio.size / app.RATE)}", "segments": []}


def start(tmp, name, brain=None):
    store = new_store(tmp, name)
    bridge = cfg_bridge.CfgBridge(store, set(), log=app.log)
    rec, tr = PressRec(), ClipTr()
    logs = os.path.join(tmp, name + "_vrc")
    os.makedirs(logs)
    path = os.path.join(logs, "output_log_2026-11-19_13-44-33.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(join(LOCAL) + T + '[Behaviour] Initialized PlayerAPI "Tester One" is local\n')
    app.clear_replies()
    stop = threading.Event()
    watcher = threading.Thread(target=app.watch_log, args=(logs, rec, tr, False, brain, store, bridge, stop), daemon=True)
    watcher.start()
    wait_for(lambda: store.speaker == "Tester One", "watcher read the backlog")
    return path, store, rec, tr, stop, watcher


def down(npc):
    return T + f"[W04AI] PTT_DOWN npc={npc} utc=1\n"


def up(npc, held=1500):
    return T + f"[W04AI] PTT_UP npc={npc} held_ms={held} utc=2\n"


def watcher_checks(tmp):
    path, store, rec, tr, stop, watcher = start(tmp, "ptt")
    try:
        # ---- review #2: two presses in a row, the first still being answered
        add(path, T + "[W04AI] NPC_STATE kwonjun=1 itoa=1\n")
        tr.gate.clear()
        add(path, down("kwonjun") + up("kwonjun"))
        wait_for(lambda: tr.started == 1, "first press reached speech-to-text")
        ok(rec.stops == 1, "the recording was taken off the microphone at release")
        add(path, down("itoa"))
        wait_for(lambda: rec.presses == 2, "second press started recording")
        add(path, up("itoa"))
        wait_for(lambda: rec.stops == 2, "second recording detached while the first is still being answered")
        tr.gate.set()
        first, second = app.replies.get(timeout=5), app.replies.get(timeout=5)
        ok(first["npc"] == "kwonjun" and first["heard"] == "clip 1", f"first press: its own audio and NPC ({first})")
        ok(second["npc"] == "itoa" and second["heard"] == "clip 2", f"second press: its own audio and NPC ({second})")
        ok(first["seq"] < second["seq"], "answered in order")

        # ---- review #11: the echo line fits the world's limit, the transcript stays whole
        long = "가나다라마바사아자차" * 15
        tr.text = long
        add(path, down("kwonjun") + up("kwonjun"))
        reply = app.replies.get(timeout=5)
        ok(len(reply["line"]) == app.WORLD_MAX_LINE and reply["line"].startswith("들은 말: 가나다") and reply["line"].endswith("…")
           and reply["heard"] == long, "a long echo is cut to 100 characters; heard keeps everything")
        tr.text = None
        short = app.make_reply("안녕", 1.0)
        ok(short["line"] == "들은 말: 안녕" and app.make_reply("", 0)["line"] == app.MISSED_LINE, "short echo lines are unchanged")
        ok(len(app.make_reply("가" * 94, 0)["line"]) == 100 and "…" not in app.make_reply("가" * 94, 0)["line"],
           "exactly 100 characters is not cut")

        # ---- review #6: one failing utterance does not end the worker
        tr.fail.append(RuntimeError("CUDA out of memory"))
        add(path, down("itoa") + up("itoa"))
        reply = app.replies.get(timeout=5)
        ok(reply["line"] == app.MISSED_LINE and reply["npc"] == "itoa", "a failed push-to-talk still closes the world's poll")
        ok(len(tsv("WORKER_ERROR")) == 1 and "RuntimeError" in tsv("WORKER_ERROR")[0], "the failure is logged")
        add(path, down("kwonjun") + up("kwonjun"))
        reply = app.replies.get(timeout=5)
        ok(reply["heard"].startswith("clip ") and reply["npc"] == "kwonjun", "the next utterance is answered")
        tr.fail.append(OSError("disk"))
        add(path, down("kwonjun") + up("kwonjun"))
        ok(app.replies.get(timeout=5)["line"] == app.MISSED_LINE, "an I/O error is survived too")
        add(path, down("kwonjun") + up("kwonjun"))
        ok(app.replies.get(timeout=5)["heard"].startswith("clip "), "and the worker keeps going")

        # ---- review #9: AFFECTION lines (a display name may have spaces)
        add(path, T + "[W04AI] AFFECTION type=pat npc=itoa by=Mikan Lover 404\n")
        itoa = store.for_npc("itoa")
        wait_for(lambda: "Mikan Lover 404" in itoa.players, "AFFECTION line noted")
        ok(itoa.players["Mikan Lover 404"]["note"] == persona_store.AFFECTION_PHRASES["pat"]
           and "Mikan Lover 404" not in store.players, "pat by a named player -> that NPC's impression of them")
        add(path, T + "[W04AI] AFFECTION type=pat npc=itoa by=Mikan Lover 404\n"
            + T + "[W04AI] AFFECTION type=hug npc=kwonjun by=Mikan Lover 404\n"
            + T + "[W04AI] AFFECTION type=bite npc=kwonjun by=Bad\n")
        wait_for(lambda: "Mikan Lover 404" in store.players, "hug for kwonjun noted")
        ok(store.players["Mikan Lover 404"]["note"] == persona_store.AFFECTION_PHRASES["hug"]
           and itoa.players["Mikan Lover 404"]["note"] == persona_store.AFFECTION_PHRASES["pat"]
           and "Bad" not in store.players, "repeat is rate-limited, other NPC separate, unknown type ignored")
        ok([d.split("\t")[-1] for d in tsv("AFFECTION")] == ["noted=1", "noted=0", "noted=1"], "AFFECTION is logged")

        # ---- NPC_POSE
        add(path, T + "[W04AI] NPC_POSE npc=itoa pose=sit\n")
        wait_for(lambda: itoa.pose == "sit", "NPC_POSE read")
        ok(store.pose == "stand", "pose is per NPC")

        # ---- review #4: a held press is dropped after the time limit ...
        stops, old_max = rec.stops, app.PTT_MAX_S
        app.PTT_MAX_S = 0.3
        try:
            add(path, down("kwonjun"))
            wait_for(lambda: rec.cancels == 1, "held too long -> cancelled")
        finally:
            app.PTT_MAX_S = old_max
        ok(tsv("PTT_CANCEL") == ["too_long"] and rec.stops == stops, "the recording is thrown away, not transcribed")
        add(path, up("kwonjun"))
        time.sleep(0.4)
        ok(app.replies.empty() and rec.stops == stops, "the late release is ignored")

        # ---- ... and when the player leaves the world with jump held
        add(path, down("kwonjun"))
        wait_for(lambda: rec.presses >= 8 and len(tsv("PTT_DOWN")) == 8, "pressed again")
        add(path, join(OTHER))
        wait_for(lambda: rec.cancels == 2, "world change -> cancelled")
        ok(tsv("PTT_CANCEL") == ["too_long", "world_changed"] and itoa.pose == "stand",
           "world change drops the recording and forgets the pose")
        add(path, join(LOCAL) + down("kwonjun") + up("kwonjun"))
        ok(app.replies.get(timeout=5)["heard"].startswith("clip "), "push-to-talk works again after coming back")
    finally:
        stop.set()
        watcher.join(timeout=3)


def recorder_checks():
    rec = object.__new__(app.Recorder)      # no sound device: only the buffer logic
    rec.lock, rec.tap, rec.rate = threading.Lock(), None, app.RATE
    rec.pre, rec.chunks, rec.recording, rec.max_chunks = collections.deque(maxlen=4), [], False, 3
    block = np.ones((512, 1), np.float32)
    rec._callback(block, 512, None, None)
    rec.start()
    for _ in range(10):
        rec._callback(block, 512, None, None)
    ok(len(rec.chunks) == 3, "a recording stops growing at its cap")
    rec.cancel()
    ok(not rec.recording and rec.chunks == [], "cancel() drops the audio without the post-roll wait")
    rec._callback(block, 512, None, None)
    ok(len(rec.pre) == 1 and rec.chunks == [], "after cancel only the pre-roll ring is fed")


# ---------------------------------------------------------------- review #10: compaction undo file

def migration_checks(tmp):
    compact = {"backups": [{"at": "2026-11-18 10:00", "notes": [{"text": "옛 메모 1", "by": "ai", "at": "x"},
                                                              {"text": "옛 메모 2", "by": "ai", "at": "x"}],
                            "after": ["합친 메모"], "players": {}}],
               "last_sig": ["합친 메모"]}

    def legacy(name):
        root = os.path.join(tmp, name)
        os.makedirs(root)
        with open(os.path.join(root, "memory.json"), "w", encoding="utf-8") as f:
            json.dump({"notes": [{"text": "합친 메모", "by": "ai", "at": "x"}]}, f, ensure_ascii=False)
        with open(os.path.join(root, "compaction.json"), "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False)
        return root

    persona = os.path.join(tmp, "mig_persona.md")
    shutil.copy(os.path.join(ROOT, "persona.md"), persona)

    # first migration: the undo backup comes along
    root = legacy("mig_first")
    st = persona_store.Store(data_dir=root, persona_path=persona)
    ok(os.path.isfile(os.path.join(root, "npcs", "kwonjun", "compaction.json"))
       and os.path.isfile(os.path.join(root, "compaction.json")) and os.path.isfile(os.path.join(root, "compaction.json.bak")),
       "compaction.json is copied on the first migration (original and .bak kept)")
    ok(st.compaction_state()["backups"] == ["2026-11-18 10:00"] and st.compaction_undo() == ""
       and [n["text"] for n in st.notes] == ["옛 메모 1", "옛 메모 2"], "the old compaction can be undone after upgrading")

    # already migrated by the older code (no compaction.json in the NPC folder): filled in
    root = legacy("mig_late")
    npc_dir = os.path.join(root, "npcs", "kwonjun")
    os.makedirs(npc_dir)
    shutil.copy(os.path.join(root, "memory.json"), os.path.join(npc_dir, "memory.json"))
    st = persona_store.Store(data_dir=root, persona_path=persona)
    ok(st.compaction_state()["backups"] == ["2026-11-18 10:00"] and st.compaction_undo() == "",
       "an already migrated store gets the missing compaction file")

    # the NPC already has its own compaction file: never overwritten
    root = legacy("mig_keep")
    npc_dir = os.path.join(root, "npcs", "kwonjun")
    os.makedirs(npc_dir)
    with open(os.path.join(npc_dir, "compaction.json"), "w", encoding="utf-8") as f:
        json.dump({"backups": [], "last_try": 5}, f)
    st = persona_store.Store(data_dir=root, persona_path=persona)
    ok(st.compaction_state()["backups"] == [] and st._compact_data().get("last_try") == 5,
       "an existing compaction file is left alone")

    # nothing legacy at all: no folder is invented; itoa never takes kwonjun's legacy file
    root = os.path.join(tmp, "mig_none")
    os.makedirs(root)
    with open(os.path.join(root, "compaction.json"), "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)
    st = persona_store.Store(data_dir=root, persona_path=persona)
    ok(not os.path.exists(os.path.join(root, "npcs", "kwonjun")), "a lone compaction.json does not start a migration")
    ok(st.for_npc("itoa").compaction_state()["backups"] == [], "itoa starts without kwonjun's backups")


# ---------------------------------------------------------------- review #14: /cfg without npc

def cfg_default_checks(tmp):
    store = new_store(tmp, "cfg")
    clock = [100.0]
    bridge = cfg_bridge.CfgBridge(store, set(), log=app.log, clock=lambda: clock[0])
    server = app.make_server(0, store, None, bridge=bridge)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        bridge.handle_line("log", LOCAL, T + "[W04AI] CFG_OPEN npc=itoa\n")
        ok(json.loads(bridge.payload())["npc"] == "ITOAR", "inside the app the last opened NPC is still the selection")
        status, body = request(port, "GET", "/cfg")
        ok(status == 200 and json.loads(body)["npc"] == "kwonjun", "GET /cfg without npc answers for kwonjun")
        status, body = request(port, "GET", "/cfg?npc=itoa")
        ok(status == 200 and json.loads(body)["npc"] == "ITOAR", "GET /cfg?npc=itoa answers for itoa")
        status, body = request(port, "GET", "/cfg?npc=../x")
        ok(status == 200 and json.loads(body)["npc"] == "kwonjun", "an unknown id answers for kwonjun")
    finally:
        server.shutdown()
        server.server_close()


def main():
    tmp = tempfile.mkdtemp(prefix="w04_review_test_")
    try:
        app.LOG_PATH = os.path.join(tmp, "voice_log.tsv")        # keep the real log clean
        step_checks(tmp)
        affection_store_checks(tmp)
        recorder_checks()
        migration_checks(tmp)
        cfg_default_checks(tmp)
        watcher_checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK: {checks} checks passed")


def test_regression_checks():
    main()


if __name__ == "__main__":
    main()
