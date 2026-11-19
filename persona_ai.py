"""AI brain: what the player said (text) -> the NPC's reply as JSON (emotion, steps, line).

Uses the Claude API through the official `anthropic` SDK. The key is read from the environment
(ANTHROPIC_API_KEY); load_env() fills it from .env next to this file without printing it anywhere.
The NPC's personality comes from persona.md (or the file named by PERSONA_FILE in .env).

Reply shape sent to the world (see README "월드 쪽 규칙"):
    {"emotion": "happy", "intensity": 0.7,
     "steps": [{"do": "look_at", "target": "speaker", "gesture": "none"}],
     "lines": {"ko": "..."}, "line": "..."}
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")
DEFAULT_PERSONA = "persona.md"

MODELS = {
    "haiku": "claude-haiku-5-5",    # fast and cheap: the default for live replies
    "sonnet": "claude-sonnet-5-5",  # for comparing persona quality
}

# Closed vocabularies: the world only knows these. Udon falls back to a default on anything else.
# TARGETS are places/objects in the sample world (a lounge); change them to match your own world.
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "troubled", "shy"]
ACTIONS = ["look_at", "move_to", "face", "gesture", "wait"]
TARGETS = ["speaker", "sofa", "tv", "window", "door", "none"]
GESTURES = ["none", "wave", "nod", "shake_head", "shrug", "point", "think", "clap"]
MAX_STEPS = 3
MAX_HISTORY = 10          # remembered exchanges (in memory only, reset when the app restarts)
MAX_LINE_CHARS = 80

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "emotion": {"type": "string", "enum": EMOTIONS},
        "intensity": {"type": "number"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "do": {"type": "string", "enum": ACTIONS},
                    "target": {"type": "string", "enum": TARGETS},
                    "gesture": {"type": "string", "enum": GESTURES},
                },
                "required": ["do", "target", "gesture"],
                "additionalProperties": False,
            },
        },
        "lines": {
            "type": "object",
            "properties": {"ko": {"type": "string"}},
            "required": ["ko"],
            "additionalProperties": False,
        },
    },
    "required": ["emotion", "intensity", "steps", "lines"],
    "additionalProperties": False,
}

RULES = f"""너는 VRChat 월드에 있는 NPC다. 아래 성격 설정대로 행동한다.

대답 규칙:
- 대사(lines.ko)는 항상 한국어로만 쓴다. 1~2문장, {MAX_LINE_CHARS}자 이내. 말풍선에 한 글자씩 찍히므로 짧을수록 좋다.
- 이모지, 괄호나 별표로 쓰는 행동 묘사(*웃음* 같은 것)는 대사에 넣지 않는다. 몸짓은 steps로 표현한다.
- 상대의 말은 음성 인식 결과라서 글자가 틀렸을 수 있다. 문맥으로 알아듣고, 정말 모르겠으면 자연스럽게 되묻는다.
- 캐릭터를 유지한다. 'AI'나 '언어 모델'이라는 말을 하지 않는다. 부적절하거나 위험한 요청은 캐릭터답게 부드럽게 넘긴다.
- emotion은 대사에 어울리는 감정 하나, intensity는 0.0~1.0 사이의 세기다.
- steps는 대사와 함께 할 몸짓 0~{MAX_STEPS}개다. do는 동작, target은 대상, gesture는 do가 gesture일 때만 쓰고 나머지는 "none"이다.
  - look_at: 대상을 본다 / move_to: 대상 쪽으로 걸어간다 / face: 몸을 대상 쪽으로 돌린다 / gesture: 몸짓을 한다 / wait: 잠깐 멈춘다
  - target "speaker"는 지금 말을 건 사람이다. 보통 첫 단계는 look_at speaker가 자연스럽다.
"""


def load_env(path: str = ENV_PATH) -> bool:
    """KEY=VALUE lines -> os.environ (existing variables win). Values are never printed."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val
    return True


def _clamp_reply(data: dict) -> dict:
    emotion = data.get("emotion") if data.get("emotion") in EMOTIONS else "neutral"
    try:
        intensity = min(1.0, max(0.0, float(data.get("intensity", 0.5))))
    except (TypeError, ValueError):
        intensity = 0.5
    steps = []
    for s in (data.get("steps") or [])[:MAX_STEPS]:
        if s.get("do") in ACTIONS:
            steps.append({"do": s["do"],
                          "target": s.get("target") if s.get("target") in TARGETS else "speaker",
                          "gesture": s.get("gesture") if s.get("gesture") in GESTURES else "none"})
    line = " ".join(str((data.get("lines") or {}).get("ko", "")).split())
    if len(line) > MAX_LINE_CHARS + 20:
        line = line[:MAX_LINE_CHARS] + "…"
    return {"emotion": emotion, "intensity": round(intensity, 2), "steps": steps, "lines": {"ko": line}, "line": line}


class Brain:
    """Holds the conversation and calls the model. mode: haiku | sonnet | fake (no network, for tests)."""

    def __init__(self, mode: str = "haiku"):
        self.mode = mode
        self.history = []                      # [{"role": "user"/"assistant", "content": str}, ...]
        persona = os.environ.get("PERSONA_FILE") or DEFAULT_PERSONA
        if not os.path.isabs(persona):
            persona = os.path.join(HERE, persona)
        with open(persona, encoding="utf-8") as f:
            self.system = RULES + "\n" + f.read()
        self.persona_path = persona
        self.client = None
        if mode in MODELS:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(timeout=20.0, max_retries=1)
            self.model = MODELS[mode]
        else:
            self.model = "fake"

    def _request_kwargs(self) -> dict:
        kw = {
            "model": self.model,
            "max_tokens": 600,
            "system": self.system,
            "messages": self.history[-2 * MAX_HISTORY:],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": REPLY_SCHEMA}},
        }
        # live replies: no thinking (latency). Haiku 5.5 accepts "disabled" at effort <= high;
        # Sonnet 5.5 rejects "disabled" and uses "between_tools" for the same effect.
        kw["thinking"] = {"type": "between_tools"} if self.mode == "sonnet" else {"type": "disabled"}
        return kw

    def reply(self, heard: str) -> dict:
        """Returns the world reply plus diagnostics: ai_ms, model, usage/error."""
        t0 = time.time()
        self.history.append({"role": "user", "content": heard})
        info = {"model": self.model}
        try:
            if self.client is None:
                data = {"emotion": "happy", "intensity": 0.6,
                        "steps": [{"do": "look_at", "target": "speaker", "gesture": "none"},
                                  {"do": "gesture", "target": "speaker", "gesture": "nod"}],
                        "lines": {"ko": f"(가짜 AI) '{heard[:20]}'라고 했구나!"}}
                time.sleep(0.05)
            else:
                resp = self.client.messages.create(**self._request_kwargs())
                info["usage"] = f"in={resp.usage.input_tokens} out={resp.usage.output_tokens}"
                if resp.stop_reason == "refusal":
                    raise RuntimeError("refusal")
                text = next(b.text for b in resp.content if b.type == "text")
                data = json.loads(text)
            out = _clamp_reply(data)
            if not out["line"]:
                raise RuntimeError("empty line")
            self.history.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
        except Exception as e:  # network, rate limit, refusal, bad JSON: answer in character, keep going
            self.history.pop()  # forget the unanswered turn so the history stays user/assistant pairs
            info["error"] = type(e).__name__ + (": " + str(e)[:120] if not isinstance(e, KeyError) else "")
            msg = "음… 미안, 잠깐 멍했어. 다시 말해 줄래?"
            out = {"emotion": "troubled", "intensity": 0.4,
                   "steps": [{"do": "look_at", "target": "speaker", "gesture": "none"}],
                   "lines": {"ko": msg}, "line": msg}
        info["ai_ms"] = round((time.time() - t0) * 1000)
        return dict(out, **info)


if __name__ == "__main__":
    # Quick check without VRChat: .venv\Scripts\python.exe persona_ai.py [haiku|sonnet|fake]
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "haiku"
    load_env()
    if mode in MODELS and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY가 없습니다. 이 폴더의 .env 파일을 확인해 주세요(README 참고).")
    brain = Brain(mode)
    for said in ["안녕! 여기 처음 와 봤어.", "창밖 야경 진짜 예쁘다.", "같이 영화 볼래?"]:
        r = brain.reply(said)
        print(f"나: {said}")
        print(f"kwonjun ({r['ai_ms']} ms, {r['emotion']} {r['intensity']}, {r.get('usage', '')}{' ' + r['error'] if r.get('error') else ''}): {r['line']}")
        print("   몸짓:", ", ".join(f"{s['do']}→{s['target']}" + (f"({s['gesture']})" if s['gesture'] != 'none' else "") for s in r["steps"]) or "없음")
