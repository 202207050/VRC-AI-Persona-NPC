"""Validated, offline settings shared by the store, brain and local Whisper."""
import re

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}
DEFAULT_MODEL = MODELS["haiku"]
MAX_HINTS = 32
MAX_HINT_CHARS = 80
MAX_HINT_TEXT = 2048

# USD per million tokens, <=100K prompt tokens, 5-minute cache. Checked 2026-11-19:
# https://platform.claude.com/docs/en/build-with-claude/prompt-caching
# The estimates compare one call: 3000 cached-prefix + 1000 other input + 200 output tokens.
MODEL_RATES = {
    MODELS["haiku"]: (0.10, 0.50, 0.125, 0.01),
    MODELS["sonnet"]: (2.0, 10.0, 2.50, 0.10),
    MODELS["opus"]: (4.0, 20.0, 5.0, 0.20),
}


# output_config.effort. Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 all accept these five levels (bundled
# claude-api shared/model-migration.md, "Choosing an effort level", read 2026-11-19). "low" is what this app
# has always sent: it is the level the docs recommend for chat, and the only one where Haiku/Sonnet answer
# without thinking first.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "low"
EFFORT_NAMES = {"low": "낮음 - 가장 빠름 (기본)", "medium": "보통 - 조금 생각하고 대답",
                "high": "높음 - 충분히 생각하고 대답", "xhigh": "매우 높음 - 느리고 비쌈",
                "max": "최대 - 가장 느리고 비쌈"}


def validate_effort(value):
    if not isinstance(value, str) or value not in EFFORTS:
        raise ValueError("생각 깊이는 low, medium, high, xhigh, max 중 하나여야 합니다.")
    return value


def effort_options():
    return [{"id": e, "name": EFFORT_NAMES[e]} for e in EFFORTS]


def validate_model(value):
    if not isinstance(value, str) or value not in MODELS.values():
        raise ValueError("지원하는 모델은 Haiku 5.5, Sonnet 5.5, Opus 5.5입니다.")
    return value


def model_options():
    options = []
    for name, model in MODELS.items():
        inp, out, write, read = MODEL_RATES[model]
        base = 1000 * inp + 200 * out
        options.append({"id": model, "name": f"Claude {name.title()} 5.5",
                        "cold_usd": round((3000 * write + base) / 1e6, 8),
                        "warm_usd": round((3000 * read + base) / 1e6, 8)})
    return options


def parse_hints(value):
    """Lines: vocabulary alone, or literal recognized text -> desired text. No regex input."""
    if not isinstance(value, str) or len(value) > MAX_HINT_TEXT:
        raise ValueError(f"닉네임 힌트는 {MAX_HINT_TEXT}자 이내의 글이어야 합니다.")
    pairs, seen = [], set()
    for line in value.splitlines():
        if not line.strip():
            continue
        parts = line.replace("→", "->").split("->")
        if len(parts) > 2:
            raise ValueError("힌트는 한 줄에 이름 하나 또는 인식된 말 -> 바꿀 말로 적어 주세요.")
        source = " ".join(parts[0].split())
        target = " ".join(parts[-1].split())
        if not source or not target or max(len(source), len(target)) > MAX_HINT_CHARS:
            raise ValueError(f"힌트 양쪽에 각각 1~{MAX_HINT_CHARS}자를 적어 주세요.")
        if source in seen:
            raise ValueError("같은 인식된 말을 두 번 적을 수 없습니다.")
        seen.add(source)
        pairs.append((source, target))
    if len(pairs) > MAX_HINTS:
        raise ValueError(f"닉네임 힌트는 최대 {MAX_HINTS}개입니다.")
    return pairs


def clean_hints(value):
    cleaned = "\n".join(a if a == b else f"{a} -> {b}" for a, b in parse_hints(value))
    if len(cleaned) > MAX_HINT_TEXT:
        raise ValueError(f"정리한 닉네임 힌트가 {MAX_HINT_TEXT}자를 넘습니다.")
    return cleaned


def hint_prompt(pairs):
    # Vocabulary, not instructions to Whisper. Each utterance gets its own NPC's snapshot.
    words = dict.fromkeys(word for pair in pairs for word in pair)
    return ", ".join(words)


def replace_hints(text, pairs):
    if not pairs:
        return text
    mapping = dict(pairs)
    # Longest match first, single pass (no cascading A->B->C). Allow Korean particles after
    # names, but never replace 404 inside 1404, 4040, user404 or 404abc. Case-sensitive.
    alternatives = "|".join(re.escape(a) for a in sorted(mapping, key=len, reverse=True))
    pattern = r"(?<!\w)(?:" + alternatives + r")(?![A-Za-z0-9_])"
    return re.sub(pattern, lambda match: mapping[match.group(0)], text)
