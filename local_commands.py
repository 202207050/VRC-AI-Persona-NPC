"""Commands understood locally before an NPC model is called."""
import re

NIGHT = re.compile(
    r"^나이트\s*모드\s*(?:밝기\s*)?(?:를\s*)?(\d{1,3})\s*(?:%|퍼센트|프로)?\s*"
    r"(?:로|으로)?\s*(?:해\s*줘|바꿔\s*줘|맞춰\s*줘|설정해\s*줘)?$"
)
MIRROR = re.compile(
    r"^거울\s*(?:을\s*)?(컷\s*아웃|cutout|고화질|저화질|hq|lq|끄|꺼|켜|off|on)"
    r"\s*(?:로|으로)?\s*(?:켜|해|바꿔|설정해)?\s*(?:줘|주세요)?$", re.IGNORECASE
)
MIRROR_MODES = {"컷아웃": "cutout", "고화질": "hq", "저화질": "lq",
                "끄": "off", "꺼": "off", "off": "off", "켜": "lq", "on": "lq"}


def world_command(text):
    text = text.strip().rstrip(".!?。！？").strip()
    match = NIGHT.fullmatch(text)
    if match and 0 <= int(match[1]) <= 100:
        return {"do": "night_mode", "action": str(int(match[1]))}
    match = MIRROR.fullmatch(text)
    if match:
        mode = "".join(match[1].lower().split())
        return {"do": "mirror", "action": MIRROR_MODES.get(mode, mode)}
    return None
