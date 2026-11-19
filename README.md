# VRC AI Persona NPC

**AI persona NPC for VRChat.** Hold the jump button, speak, and the NPC answers in character with a line, an emotion and gestures. Speech-to-text runs on your PC (Whisper), and the replies come from Claude.

VRChat 월드의 NPC에 AI 인격을 붙이는 PC 앱입니다. 점프 버튼을 누른 채 말하면 NPC가 성격대로 대답하고, 감정과 몸짓도 함께 고릅니다.

> **지금 상태 (2026-11):** 시험 단계입니다. 이 앱은 아래 "월드 쪽 규칙"을 지키는 VRChat 월드에서만 동작합니다. 예제 월드는 아직 공개 전입니다.

---

## 어떻게 동작하나요

```
[VRChat 월드]                          [내 PC: 이 앱]                         [Claude API]
점프 누름 ──기록 파일에 PTT_DOWN──▶  마이크 녹음 시작
          ──localhost 요청 보내고 대기─▶
점프 뗌   ──기록 파일에 PTT_UP────▶  녹음 끝 → Whisper로 글자 변환
                                      (그래픽카드로 약 0.1초)
                                      글자 ─────────────────────────────▶  성격대로 대답 고르기
                                      ◀──── 대사·감정·몸짓 (JSON) ────────  (Haiku 기준 1~2.5초)
NPC가 말함 ◀──── 기다리던 요청에 답 ──
```

- VRChat 월드는 마이크 소리를 직접 읽을 수 없습니다. 그래서 이 앱이 내 PC에서 대신 듣습니다.
- 앱은 VRChat 기록 파일(`output_log`)에 찍히는 점프 표시를 보고 녹음을 시작하고 멈춥니다.
- 대답은 월드가 내 PC 주소(`http://127.0.0.1:18770/poll`)로 보낸 요청에 대한 답으로 전달됩니다.
- 앱을 켠 사람만 NPC에게 말할 수 있습니다. 같은 방의 다른 사람도 NPC의 대답은 볼 수 있지만, 이것은 월드 쪽에서 동기화해야 합니다.

## 준비물

| 무엇 | 설명 |
|---|---|
| Windows 10/11 + VRChat(PC) | 데스크톱과 PC VR 모두 됩니다. |
| Python 3.10 이상 | [python.org](https://www.python.org/downloads/)에서 설치합니다. 설치 화면의 "py launcher"를 켜 두세요. |
| NVIDIA 그래픽카드 (권장) | RTX 20 시리즈 이상을 권장합니다. 없으면 CPU로 동작하지만 글자 변환이 몇 초 걸립니다. |
| Claude API 키 (유료) | [Claude Console](https://console.anthropic.com)에서 만듭니다. 사용한 만큼 돈이 나갑니다(아래 "비용" 참고). |
| 디스크 약 5 GB | PyTorch 2~3 GB + Whisper 모델 1.6 GB |

## 설치 (처음 한 번)

1. **받기:** 이 페이지 위쪽의 초록색 **Code** 버튼 → **Download ZIP**으로 받아 원하는 폴더에 풉니다. git을 쓴다면 `git clone`으로 받아도 됩니다.
2. **설치:** 폴더 안의 `setup.cmd`를 더블클릭합니다.
   - 가상환경(`.venv`)을 만들고 PyTorch와 필요한 패키지를 받습니다. 몇 분 걸립니다.
   - NVIDIA 그래픽카드가 있으면 그래픽카드용 PyTorch를, 없으면 CPU용을 받습니다.
   - GTX 10 시리즈처럼 오래된 그래픽카드는 명령창에서 `setup.cmd cu126`으로 설치하세요.
3. **API 키 넣기:** 설치가 끝나면 같은 폴더에 `.env` 파일이 생깁니다. 메모장이나 VS Code로 열어서 `ANTHROPIC_API_KEY=` 바로 뒤에 키를 붙여 넣고 저장합니다.
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   - `.env`는 내 PC에만 두세요. 키를 채팅, 월드, GitHub 등 남이 볼 수 있는 곳에 올리면 안 됩니다. 이 저장소의 `.gitignore`는 `.env`를 업로드 대상에서 빼 둡니다.
   - 키 없이 켜면 AI 대신 "들은 말: …"로 따라 말하는 시험 모드로 동작합니다.

## 켜기

1. `run.cmd`를 더블클릭합니다.
2. 처음 한 번은 Whisper 모델(약 1.6 GB)을 받습니다. 그다음부터는 10~20초 뒤에 이렇게 나옵니다.
   ```
   준비 완료. 마이크: ... / 계산: cuda / 답하는 쪽: AI claude-haiku-5-5 / ...
   월드 연결 대기: http://127.0.0.1:18770/poll
   ```
   계산이 `cuda`면 그래픽카드, `cpu`면 CPU로 계산합니다.
3. VRChat에서 이 규칙을 지키는 월드에 들어갑니다.
   - 월드가 내 PC 주소로 요청하려면 VRChat 설정의 **Allow Untrusted URLs**가 필요할 수 있습니다(확인 중).
4. **점프(데스크톱 Space / VR 오른손 A)를 누른 채 말하고, 다 말하면 뗍니다.** NPC가 생각하는 표시를 띄운 뒤 대답합니다.
5. 끌 때는 앱 창을 닫습니다.

앱 창에는 들은 말과 대답이 시간과 함께 나옵니다.
```
12:00:01 녹음 시작
12:00:03 녹음 끝, 글자로 바꾸는 중...
12:00:03 들음 (110 ms): 창밖 야경 진짜 예쁘다.
12:00:05 kwonjun (1.8초, happy): 그치? 난 저 야경 보는 게 제일 좋아.
```

## 마이크 고르기

앱은 Windows 기본 마이크를 씁니다. VR에서는 VRChat이 헤드셋 마이크(Virtual Desktop, Oculus 등)를 쓰는 경우가 많으니 같은 마이크를 골라 주세요.

1. 명령창에서 `run.cmd --list`를 실행하면 마이크 목록과 번호가 나옵니다.
2. 쓸 번호를 `.env`에 적습니다. 예: `MIC_DEVICE=3`. 한 번만 쓸 때는 `run.cmd --device 3`으로 켜도 됩니다.

## 성격 바꾸기

- **`persona.md`가 NPC의 인격입니다.** 이름, 말투, 좋아하는 것, 주변에 있는 물건을 자유롭게 고치고 앱을 다시 켜면 적용됩니다. 예시로 `kwonjun`이 들어 있습니다.
- 다른 파일을 쓰려면 `.env`에 `PERSONA_FILE=내캐릭터.md`를 적습니다.
- 대답 규칙(한국어로 짧게, 이모지 금지 등)과 고를 수 있는 감정·몸짓 목록은 `persona_ai.py`의 `RULES`, `EMOTIONS`, `ACTIONS`, `TARGETS`, `GESTURES`에 있습니다. `TARGETS`(소파, TV, 창문, 문)는 예제 월드의 물건이니 내 월드에 맞게 바꾸세요.
- 대화는 앱이 켜져 있는 동안 최근 10번만 기억합니다. 앱을 다시 켜면 처음부터 시작합니다.
- Claude Sonnet으로 대답하게 하려면 `run.cmd --ai sonnet`으로 켭니다. 더 느리고 비싸지만 성격 표현이 더 풍부할 수 있습니다.
- VRChat 없이 대답만 시험하려면 명령창에서 `.venv\Scripts\python.exe persona_ai.py`를 실행합니다. 시험 문장 3개를 보내고 대답을 보여 줍니다.

## 비용

대답 한 번에 성격 설정과 최근 대화가 함께 보내져서 입력이 약 2,000토큰, 출력이 약 150토큰입니다.

| 모델 | 요금(100만 토큰당 입력 / 출력) | 대답 한 번 |
|---|---|---|
| Claude Haiku 5.5 (기본) | $0.10 / $0.50 | 약 $0.0003 |
| Claude Sonnet 5.5 (`--ai sonnet`) | $2 / $10 | 약 $0.0055 |

요금은 바뀔 수 있으니 [Claude Console](https://console.anthropic.com)의 요금 안내와 사용량을 확인하세요. Console의 **Billing** 페이지에서 월 사용 한도를 낮게 걸어 둘 수 있습니다([안내](https://platform.claude.com/docs/api/rate-limits)).

## 개인정보

- 마이크는 앱이 켜져 있는 동안 열려 있지만, 메모리에는 **최근 0.5초만** 들고 있다가 계속 버립니다. 점프를 누르는 순간 첫 소리가 잘리지 않게 하려는 것입니다.
- 녹음은 점프를 누른 동안(+뗀 뒤 0.25초)만 하고 **파일로 남기지 않습니다.** `--save-clips`로 켤 때만 `clips/` 폴더에 저장됩니다.
- 소리는 내 PC에서 글자로 바뀌고, **Claude API에는 글자만** 보내집니다.
- 들은 말과 대답은 내 PC의 `voice_log.tsv`에 기록됩니다. 필요 없으면 지워도 됩니다.

## 월드 쪽 규칙 (월드 제작자용)

이 앱과 함께 동작하려면 월드(Udon)가 아래처럼 하면 됩니다.

1. 점프를 누르면 기록에 `[W04AI] PTT_DOWN utc=<DateTime.UtcNow.Ticks>`를 남깁니다(`Debug.Log`). 그리고 미리 넣어 둔 `VRCUrl` `http://127.0.0.1:18770/poll`로 String Loading을 시작합니다.
2. 점프를 떼면 `[W04AI] PTT_UP held_ms=<누른 시간 ms> utc=<ticks>`를 남깁니다. 0.35초보다 짧으면 앱이 실수로 누른 것으로 봅니다.
3. 앱은 대답이 준비되면 그 요청에 JSON으로 답합니다. 55초 안에 대답이 없으면 `{}`를 보냅니다.
   ```json
   {"emotion": "happy", "intensity": 0.7,
    "steps": [{"do": "look_at", "target": "speaker", "gesture": "none"},
              {"do": "gesture", "target": "speaker", "gesture": "wave"}],
    "lines": {"ko": "어서 와!"}, "line": "어서 와!",
    "heard": "안녕", "stt_ms": 110, "ai_ms": 1800, "seq": 3}
   ```
   - `line`: NPC가 말할 대사. 못 알아들었을 때는 "응? 잘 못 들었어."가 옵니다.
   - `emotion`: `neutral, happy, sad, angry, surprised, troubled, shy` 중 하나. `intensity`는 0~1입니다.
   - `steps`: 0~3개. `do`는 `look_at, move_to, face, gesture, wait`, `target`은 `speaker`(말을 건 사람)와 월드 물건 이름, `gesture`는 `none, wave, nod, shake_head, shrug, point, think, clap`입니다.
4. 참고: VRChat은 String Loading을 5초에 한 번으로 제한합니다. Untrusted URL 설정이 필요할 수 있습니다.

## 문제 해결

| 증상 | 이유와 해결 |
|---|---|
| "API 키가 없어서 … 따라 말하기로 동작" | `.env`에 키가 없거나 줄 모양이 틀렸습니다. `ANTHROPIC_API_KEY=키` 한 줄을 띄어쓰기 없이 넣고 다시 켜세요. |
| "CPU로 계산합니다" | 그래픽카드를 못 찾았거나 드라이버가 오래됐습니다. NVIDIA 드라이버를 최신으로 올리거나, 오래된 그래픽카드면 `setup.cmd cu126`으로 다시 설치하세요. |
| 점프를 눌러도 "녹음 시작"이 안 나옴 | 이 규칙을 지키는 월드가 아니거나, 앱이 다른 기록 파일을 보고 있습니다. 앱 창의 "VRChat 기록 감시 중:" 파일 이름을 확인하세요. |
| 녹음은 되는데 NPC가 대답을 안 함 | 월드가 내 PC 주소에 요청하지 못하고 있습니다. VRChat의 Allow Untrusted URLs를 켜 보세요. 다른 프로그램이 18770 포트를 쓰고 있지 않은지도 확인하세요. |
| 말하지 않았는데 이상한 문장이 나옴 | Whisper는 조용한 소리에서 "시청해 주셔서 감사합니다" 같은 말을 지어내는 버릇이 있습니다. 자주 나오는 말은 앱이 걸러 내지만, 새로 보이면 `persona_npc_app.py`의 `HALLUCINATIONS`에 더하세요. |
| 내 목소리가 아닌 소리를 들음 | 다른 마이크를 듣고 있습니다. "마이크 고르기"를 보세요. |

## 파일

| 파일 | 내용 |
|---|---|
| `setup.cmd` / `run.cmd` | 설치 / 켜기 |
| `persona_npc_app.py` | 기록 감시, 녹음, Whisper, 월드와 주고받기 |
| `persona_ai.py` | Claude에 묻고 대답(JSON)을 받기 |
| `persona.md` | NPC 성격(예시: kwonjun) |
| `.env.example` | 설정 파일 샘플. 복사해서 `.env`로 씁니다. |
| `requirements.txt` | 파이썬 패키지 목록 |

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 보세요.

이 앱이 설치해서 쓰는 것들은 각자의 라이선스를 따릅니다: [OpenAI Whisper](https://github.com/openai/whisper)(MIT, 모델 포함), [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python), [PyTorch](https://pytorch.org)(BSD), [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)(MIT).
