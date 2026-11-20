"""Offline check of the reply effort setting, "think harder when needed" (recall) and memo selection.

    .venv\\Scripts\\python.exe -X utf8 tests\\test_recall.py

No API call, no network: the Claude client is a fake that returns queued answers; files go to a temp folder.
"""
import json
import os
import shutil
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("PERSONA_FILE", None)
os.environ["W04_WEB_SEARCH"] = "1"

import cfg_bridge      # noqa: E402
import persona_ai      # noqa: E402
import persona_store   # noqa: E402
from voice_options import EFFORTS, MODELS  # noqa: E402

NS = types.SimpleNamespace
checks = 0


def ok(cond, what):
    global checks
    if not cond:
        raise AssertionError(what)
    checks += 1


def answer(line, **extra):
    return dict({"emotion": "happy", "intensity": 0.5, "steps": [], "lines": {"ko": line}}, **extra)


class FakeMessages:
    def __init__(self):
        self.requests, self.answers = [], []

    def create(self, **kw):
        self.requests.append(kw)
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return NS(stop_reason="end_turn",
                  content=[NS(type="thinking", thinking=""), NS(type="text", text=json.dumps(item, ensure_ascii=False))],
                  usage=NS(input_tokens=10, output_tokens=5))


def system_text(req):
    return "\n".join(b["text"] for b in req["system"])


def props(req):
    return req["output_config"]["format"]["schema"]["properties"]


def main():
    tmp = tempfile.mkdtemp(prefix="w04_recall_test_")
    try:
        persona = os.path.join(tmp, "persona.md")
        shutil.copy(os.path.join(ROOT, "persona.md"), persona)
        store = persona_store.Store(data_dir=os.path.join(tmp, "data"), persona_path=persona)
        store.set_speaker("Alice")
        brain = persona_ai.Brain("fake", store)
        fake = FakeMessages()
        brain.client, brain.mode = NS(messages=fake), "haiku"     # real request path, fake network

        # ---- effort setting: defaults keep today's behaviour
        ok(store.settings["brain_effort"] == "low" and store.settings["auto_recall"] is True, "defaults: low effort, recall on")
        ok([o["id"] for o in store.state()["effort_options"]] == list(EFFORTS) == ["low", "medium", "high", "xhigh", "max"],
           "the page offers the five documented levels")
        fake.answers.append(answer("안녕!"))
        brain.reply("안녕")
        req = fake.requests[-1]
        ok(req["output_config"]["effort"] == "low" and req["thinking"] == {"type": "disabled"}
           and req["max_tokens"] == 600 and "timeout" not in req, "low: thinking off, 600 tokens, default timeout (as before)")

        # ---- effort x model: only settings each model accepts
        adaptive = {"type": "adaptive", "display": "omitted"}
        expect = {
            ("haiku", "low"): {"type": "disabled"}, ("sonnet", "low"): {"type": "between_tools"}, ("opus", "low"): adaptive,
        }
        for name in ("haiku", "sonnet", "opus"):
            store.apply({"settings": {"brain_model": MODELS[name]}})
            for effort in EFFORTS:
                store.apply({"settings": {"brain_effort": effort}})
                fake.answers.append(answer("응"))
                r = brain.reply("테스트")
                req = fake.requests[-1]
                ok("error" not in r and r["line"] == "응", f"{name}/{effort}: answer read past the thinking block")
                ok(req["model"] == MODELS[name] and req["output_config"]["effort"] == effort, f"{name}/{effort}: effort sent")
                ok(req["thinking"] == expect.get((name, effort), adaptive), f"{name}/{effort}: thinking setting")
                ok(req["thinking"]["type"] != "disabled" or (name == "haiku" and effort == "low"),
                   "thinking is never 'disabled' on Sonnet/Opus or above low")
                if effort != "low":
                    ok(req["max_tokens"] >= 4096 and req["timeout"] > 20, f"{name}/{effort}: room and time for thinking")
        store.apply({"settings": {"brain_model": MODELS["haiku"], "brain_effort": "low"}})
        for bad in ("extreme", "", None, 3):
            try:
                store.apply({"settings": {"brain_effort": bad}})
                ok(False, f"bad effort accepted: {bad!r}")
            except ValueError:
                ok(True, "bad effort refused")
        store.apply({"settings": {"brain_effort": "medium"}})
        ok(persona_store.Store(data_dir=os.path.join(tmp, "data"), persona_path=persona).settings["brain_effort"] == "medium",
           "effort survives a restart")
        store.apply({"settings": {"brain_effort": "low"}})
        bridge = cfg_bridge.CfgBridge(store)
        ok(bridge.apply({"op": "effort", "effort": "high"}) == "" and store.settings["brain_effort"] == "high", "cfg op: effort")
        ok(bridge.apply({"op": "effort", "effort": "turbo"}) != "" and store.settings["brain_effort"] == "high", "cfg op: bad effort refused")
        ok(bridge.apply({"op": "recall", "on": False}) == "" and store.settings["auto_recall"] is False, "cfg op: recall off")
        store.apply({"settings": {"brain_effort": "low", "auto_recall": True}})
        store.apply({"history_clear": True})

        # ---- the normal call offers "recall"; ordinary talk stays one call
        fake.requests.clear()
        fake.answers.append(answer("안녕!"))
        r = brain.reply("안녕")
        ok(len(fake.requests) == 1 and "recall" not in r, "small talk: one call")
        ok(list(props(fake.requests[0]))[:2] == ["search", "recall"] and props(fake.requests[0])["recall"] == {"type": "boolean"},
           "schema offers search and recall first")
        ok("recall" not in fake.requests[0]["output_config"]["format"]["schema"]["required"], "recall is optional")
        ok("recall(기억 되짚기)" in fake.requests[0]["system"][0]["text"], "the recall rule sits in the cached fixed block")

        # ---- a recall turn: interim line, then one careful call with every memo
        notes = [{"text": f"잡담 메모 {i}: 날씨 이야기를 함", "by": "ai"} for i in range(30)]
        notes[3] = {"text": "Alice는 다음 주 토요일에 같이 영화 보기로 약속했다", "by": "ai"}
        notes[5] = {"text": "Alice는 귤을 좋아하고 커피는 싫어한다", "by": "ai"}
        notes[7] = {"text": "절대 잊지 말 것: 고양이 이름은 보리", "by": "user"}
        store.apply({"notes": notes})
        fake.requests.clear()
        interim = []
        fake.answers += [answer("음… 잠깐 생각해 볼게.", recall=True, impression="미리 쓴 인상"),
                         answer("저번에 토요일에 같이 영화 보자고 했잖아!", memory_note="Alice가 약속을 확인함")]
        r = brain.reply("저번에 내가 뭐 하자고 했었지?", on_interim=interim.append)
        ok(len(fake.requests) == 2 and r["line"] == "저번에 토요일에 같이 영화 보자고 했잖아!" and "error" not in r,
           "recall turn: two calls, the second answer is the reply")
        ok(len(interim) == 1 and interim[0]["line"] == "음… 잠깐 생각해 볼게." and interim[0]["interim"] == "recall",
           "the waiting line goes out first through on_interim")
        ok(r["recall"] is True and r["recall_ms"] >= 0 and "recall" not in r["lines"], "diagnostics, world reply shape unchanged")
        one, two = fake.requests
        ok(one["output_config"]["effort"] == "low" and one["thinking"] == {"type": "disabled"}, "first call stays fast")
        ok(two["output_config"]["effort"] == "high" and two["thinking"] == adaptive and two["max_tokens"] >= 4096
           and two["timeout"] > 20, "second call: higher effort with thinking, more room and time")
        ok("recall" not in props(two) and "search" not in props(two) and "tools" not in two, "second call cannot recall or search again")
        ok(two["messages"][-1]["content"].startswith("저번에 내가 뭐 하자고 했었지?") and "기억을 되짚어" in two["messages"][-1]["content"],
           "second call is told to look through its memory")
        ok(all(n["text"] in system_text(two) for n in notes), "second call sees every memo")
        ok(one["system"][0] == two["system"][0], "the cached fixed block is identical in both calls")
        ok(store.history[-2]["content"] == "저번에 내가 뭐 하자고 했었지?" and len(store.history) == 4,
           "history keeps only the player's own words, one exchange")
        ok(store.notes[-1]["text"] == "Alice가 약속을 확인함" and store.players["Alice"]["note"] != "미리 쓴 인상",
           "memory updates come from the final answer only")

        # ---- the NPC's own higher effort wins over the recall level
        store.apply({"settings": {"brain_effort": "max"}})
        fake.requests.clear()
        fake.answers += [answer("잠깐만.", recall=True), answer("기억났어!")]
        brain.reply("내가 좋아하는 과일이 뭐였지?")
        ok([q["output_config"]["effort"] for q in fake.requests] == ["max", "max"], "effort max: the recall call is not lowered to high")
        store.apply({"settings": {"brain_effort": "low"}})

        # ---- search wins when both are asked; a failed recall call answers in character
        fake.requests.clear()
        fake.answers += [answer("찾아볼게!", search="서울 날씨", recall=True), answer("맑대!")]
        interim.clear()
        r = brain.reply("오늘 날씨 어때?", on_interim=interim.append)
        ok(r.get("search_query") == "서울 날씨" and "recall" not in r and "tools" in fake.requests[1]
           and interim[0]["interim"] == "search", "search and recall together: only the search runs")
        n = len(store.history)
        fake.answers += [answer("음…", recall=True), RuntimeError("timeout")]
        r = brain.reply("저번 약속 기억나?")
        ok("error" in r and r["line"] and len(store.history) == n, "failed recall call: in-character apology, history untouched")

        # ---- memo selection in the normal call (no extra model call)
        fake.requests.clear()
        fake.answers.append(answer("토요일이지!"))
        brain.reply("우리 영화 보기로 약속한 게 무슨 요일이었지?")
        ok(len(fake.requests) == 1, "selection is local: still one call")
        sent = system_text(fake.requests[0])
        ok("Alice는 다음 주 토요일에 같이 영화 보기로 약속했다" in sent, "the memo that matches the question is sent")
        ok("절대 잊지 말 것: 고양이 이름은 보리" in sent, "hand-written memos are sent")
        ok(store.notes[-1]["text"] in sent and store.notes[-2]["text"] in sent, "the newest memos are sent")
        listed = [ln for ln in sent.split("## 오래 기억하고 있는 것")[1].split("## ")[0].splitlines() if ln.startswith("- ")]
        ok(len(listed) - 1 <= persona_store.PICK_RELEVANT + persona_store.PICK_RECENT + persona_store.PICK_USER
           and len(listed) - 1 < len(store.notes), "only a part of the memos is sent")
        ok("개 더 있다" in listed[-1] and "recall" in listed[-1], "the AI is told there are more memos and how to reach them")
        store.apply({"history_clear": True})      # the previous thing the player said also counts (follow-ups)
        picked = store.pick_notes("귤 먹을래?", "Alice")
        ok(any("귤을 좋아하고" in n["text"] for n in picked) and not any("영화 보기로" in n["text"] for n in picked),
           "another question picks other memos")
        order = [store.notes.index(n) for n in picked]
        ok(order == sorted(order), "picked memos keep their stored order")
        ok("Alice는 귤을 좋아하고" in brain.system_prompt(), "the settings page preview still shows every memo")

        # few memos: all of them, no hint line
        few = persona_store.Store(data_dir=os.path.join(tmp, "few"), persona_path=persona)
        few.apply({"notes": [{"text": f"메모 {i}"} for i in range(persona_store.PICK_ALL_UNDER)]})
        ok(len(few.pick_notes("아무 말")) == persona_store.PICK_ALL_UNDER and "더 있다" not in few.prompt_tail("Alice", "아무 말"),
           "12 memos or fewer: all are sent")

        # ---- recall off: no field, no rule, every memo in the normal call (as before)
        store.apply({"settings": {"auto_recall": False}})
        fake.requests.clear()
        fake.answers.append(answer("응", recall=True))
        r = brain.reply("저번에 뭐라 했지?")
        ok(len(fake.requests) == 1 and "recall" not in r and r["line"] == "응", "recall off: one call even if the model asks")
        ok("recall" not in props(fake.requests[0]) and "recall(기억 되짚기)" not in system_text(fake.requests[0]), "no field, no rule")
        ok(all(n["text"] in system_text(fake.requests[0]) for n in store.notes) and "더 있다" not in system_text(fake.requests[0]),
           "recall off: all memos are sent")

        # ---- the offline brain (--ai fake / off) never recalls
        offline = persona_ai.Brain("fake", persona_store.Store(data_dir=os.path.join(tmp, "off"), persona_path=persona))
        ok(offline.recall_on() is False and "recall(기억 되짚기)" not in offline.system_prompt(), "offline brain: no recall")
        print(f"OK: {checks} checks passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
