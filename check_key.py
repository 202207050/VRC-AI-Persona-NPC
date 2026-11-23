"""API key check for first-time users: where the key comes from, whether it looks right, and
whether Claude accepts it. The key itself is never printed (only its first 10 and last 4 characters)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Anthropic SDK settings that change where the key goes or replace it. If any is set in Windows,
# a correct key in .env can still fail.
SDK_VARS = ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
            "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN",
            "ANTHROPIC_FEDERATION_RULE_ID"]


def mask(key: str) -> str:
    return f"{key[:10]}…{key[-4:]} (길이 {len(key)})" if len(key) > 16 else f"(너무 짧음, 길이 {len(key)})"


def main() -> int:
    from_windows = bool(os.environ.get("ANTHROPIC_API_KEY"))
    from persona_ai import ENV_PATH, load_env
    has_env = load_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    problems = []

    print("== API 키 확인 ==")
    print(f".env 파일: {'있음' if has_env else '없음'} ({ENV_PATH})")
    if not key:
        print("키: 없음")
        problems.append(".env의 ANTHROPIC_API_KEY= 뒤에 키가 없습니다. setup.cmd를 다시 실행해 키를 넣으세요.")
    else:
        print(f"키를 가져온 곳: {'Windows 환경 변수 (.env보다 먼저 쓰임!)' if from_windows else '.env 파일'}")
        print(f"키 모양: {mask(key)}")
        if from_windows:
            problems.append("Windows 환경 변수에 ANTHROPIC_API_KEY가 따로 있어서 .env의 키는 무시됩니다. "
                            "'시스템 환경 변수 편집 → 환경 변수'에서 ANTHROPIC_API_KEY를 지우거나 새 키로 바꾸세요.")
        if not key.startswith("sk-ant-"):
            problems.append("키가 sk-ant-로 시작하지 않습니다. Console의 API Keys에서 만든 키 전체를 복사하세요.")
        if any(ch.isspace() for ch in key) or any(ord(ch) > 126 for ch in key):
            problems.append("키 안에 띄어쓰기나 한글 같은 다른 글자가 섞여 있습니다. 키만 다시 붙여 넣으세요.")
    official = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/") == "https://api.anthropic.com"
    other = [name for name in SDK_VARS if os.environ.get(name) and not (name == "ANTHROPIC_BASE_URL" and official)]
    if other:
        print("키 외의 Anthropic 설정: " + ", ".join(other))
        problems.append("Windows에 " + ", ".join(other) + " 설정이 있습니다. 다른 프로그램용 설정이면 이 앱의 "
                        "요청이 엉뚱한 곳으로 가서 키가 틀렸다고 나올 수 있습니다. 환경 변수에서 지우고 다시 시험하세요.")
    if os.environ.get("ANTHROPIC_BASE_URL") and not official:
        print(f"  ANTHROPIC_BASE_URL = {os.environ['ANTHROPIC_BASE_URL']}")

    if key:
        print("Claude에 키를 보내 확인하는 중... (요금 없음)")
        import anthropic
        try:
            anthropic.Anthropic(timeout=20.0, max_retries=0).models.list(limit=1)
            print("결과: 키가 정상입니다. ✔")
        except anthropic.AuthenticationError:
            print("결과: Claude가 이 키를 거절했습니다 (401 키 틀림).")
            problems.append("Console(console.anthropic.com)의 API Keys에서 키를 새로 만들어 setup.cmd로 다시 넣으세요. "
                            "키를 지웠거나, 복사가 덜 됐거나, 다른 계정의 키일 수 있습니다.")
        except anthropic.PermissionDeniedError:
            print("결과: 키는 맞지만 권한이 없습니다 (403).")
            problems.append("Console에서 키의 워크스페이스 권한과 결제(Billing)를 확인하세요.")
        except Exception as e:  # network, proxy, firewall
            print(f"결과: 확인하지 못했습니다 ({type(e).__name__}).")
            problems.append("인터넷 연결이나 방화벽·백신이 막고 있는지 확인하세요.")

    print()
    if problems:
        print("해야 할 일:")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        print("고친 뒤에는 앱 창을 닫고 다시 켜세요(앱은 켤 때 한 번만 키를 읽습니다).")
        return 1
    print("문제 없음. 앱이 이미 켜져 있었다면 닫고 다시 켜세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
