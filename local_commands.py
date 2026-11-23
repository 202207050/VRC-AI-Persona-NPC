"""Commands understood locally before an NPC model is called."""
import re

PREFIX = r"(?:(?P<who>{extra}전원|모두|다들|다\s*같이|전부|나)\s*(?:한테|에게|도|다)?\s*)?"
NIGHT = re.compile(
    "^" + PREFIX.format(extra="") + r"(?:나이트|야간|밤)\s*모드\s*(?:밝기\s*)?(?:를\s*)?(?P<pct>\d{1,3})\s*(?:%|퍼센트|프로)?\s*"
    r"(?:로|으로)?\s*(?:해\s*줘|바꿔\s*줘|맞춰\s*줘|설정해\s*줘)?$"
)
MIRROR = re.compile(
    "^" + PREFIX.format(extra=r"여기\s*있는\s*사람들|내|")
    + r"(?:거울|미러)\s*(?:을|를)?\s*(?P<mode>컷\s*아웃|cutout|고화질|저화질|hq|lq|끄|꺼|켜|off|on)"
    r"\s*(?:로|으로)?\s*(?:켜|해|바꿔|설정해)?\s*(?:줘|주세요)?$", re.IGNORECASE
)
MIRROR_MODES = {"컷아웃": "cutout", "고화질": "hq", "저화질": "lq",
                "끄": "off", "꺼": "off", "off": "off", "켜": "lq", "on": "lq"}


def world_command(text):
    text = text.strip().rstrip(".!?。！？").strip()
    match = NIGHT.fullmatch(text)
    if match and 0 <= int(match["pct"]) <= 100:
        command = {"do": "night_mode", "action": str(int(match["pct"]))}
    else:
        match = MIRROR.fullmatch(text)
        if not match:
            return None
        mode = "".join(match["mode"].lower().split())
        command = {"do": "mirror", "action": MIRROR_MODES.get(mode, mode)}
    if match["who"] and match["who"] not in ("나", "내"):
        command["who" if command["do"] == "mirror" else "scope"] = "all"
    return command
