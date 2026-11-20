# VRC AI Persona NPC

**AI persona NPC for VRChat.** Look at the NPC from close by and just talk (or hold the jump button), and the NPC answers in character with a line, an emotion and gestures. Speech-to-text runs on your PC (Whisper), and the replies come from Claude.

VRChat 월드의 NPC에 AI 인격을 붙이는 PC 앱입니다. NPC를 가까이서 바라보고 그냥 말하면(또는 점프 버튼을 누른 채 말하면) NPC가 성격대로 대답하고, 감정과 몸짓도 함께 고릅니다.

두 NPC(`kwonjun`, `ITOAR`)를 따로 켜고 끌 수 있고, 바라본 NPC가 대답합니다. NPC마다 성격·사람 인상·기억이 따로 저장되고, 설정 화면에서 NPC를 골라 고칠 수 있습니다. 두 NPC가 함께 있으면 가끔 서로 짧은 잡담도 합니다.

> **지금 상태 (2026-11):** 시험 단계입니다. 이 앱은 아래 "월드 쪽 규칙"을 지키는 VRChat 월드에서만 동작합니다. 예제 월드는 아직 공개 전입니다.

---

## 어떻게 동작하나요

```
[VRChat 월드]                          [내 PC: 이 앱]                         [Claude API]
NPC를 가까이서 바라봄
          ──기록 파일에 LISTEN_ON──▶  마이크 듣기 시작 (말소리가 나면 녹음)
          ──localhost 요청 보내고 대기─▶
말을 마침 (0.8초 조용함)              녹음 끝 → Whisper로 글자 변환
                                      (그래픽카드로 약 0.1초)
                                      글자 ─────────────────────────────▶  성격대로 대답 고르기
                                      ◀──── 대사·감정·몸짓 (JSON) ────────  (Haiku 기준 1~2.5초)
NPC가 말함 ◀──── 기다리던 요청에 답 ──
```

- VRChat 월드는 마이크 소리를 직접 읽을 수 없습니다. 그래서 이 앱이 내 PC에서 대신 듣습니다.
- 앱은 VRChat 기록 파일(`output_log`)에 찍히는 "듣기 시작/끝" 표시(또는 점프 표시)를 보고 마이크를 듣습니다. 자세한 내용은 아래 "말 거는 방법"에 있습니다.
- 대답은 월드가 내 PC 주소(`http://127.0.0.1:18770/poll`)로 보낸 요청에 대한 답으로 전달됩니다.
  앱은 경로가 정확히 `/poll`이고 주소가 `127.0.0.1` 또는 `localhost`인 요청에만 대답을 내줍니다. 웹 페이지가 보낸 요청(`Origin`이 붙거나 다른 사이트에서 온 것)과 다른 이름으로 접속한 요청은 403으로 거절하고 대답을 꺼내지 않습니다.
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
   준비 완료. 마이크: ... / 계산: cuda / 답하는 쪽: NPC별 설정 모델 / ...
   월드 연결 대기: http://127.0.0.1:18770/poll
   ```
   계산이 `cuda`면 그래픽카드, `cpu`면 CPU로 계산합니다.
3. VRChat에서 이 규칙을 지키는 월드에 들어갑니다.
   - 월드가 내 PC 주소로 요청하려면 VRChat 설정의 **Allow Untrusted URLs**가 필요할 수 있습니다(확인 중).
4. **NPC에게 가까이 가서(약 2.5 m 안) NPC를 화면 가운데쯤에 두고 그냥 말합니다.** VRChat 마이크처럼 버튼을 누를 필요가 없습니다. 말을 마치고 0.8초쯤 지나면 NPC가 대답합니다.
   - 예전처럼 점프(데스크톱 Space / VR 오른손 A)를 누른 채 말하고 떼는 방법은 월드에서 점프 말하기(W04Probe의 `jumpTalk`)를 켠 경우에만 됩니다. 예제 월드의 기본값은 꺼짐입니다(점프는 그냥 점프).
5. 끌 때는 앱 창을 닫습니다.

앱 창에는 들은 말과 대답이 시간과 함께 나옵니다.
```
12:00:00 NPC가 듣는 중 (그냥 말하면 됩니다)
12:00:03 말 끝남, 글자로 바꾸는 중...
12:00:03 들음 (110 ms): 창밖 야경 진짜 예쁘다.
12:00:05 kwonjun (1.8초, happy): 그치? 난 저 야경 보는 게 제일 좋아.
12:00:20 듣기 멈춤 (NPC를 안 보거나 멀어짐)
```

## 말 거는 방법 (바라보고 말하기 / 점프로 말하기)

기본은 **바라보고 말하기**(gaze)입니다.

- **언제 듣나요:** NPC가 내 시야 가운데에서 약 15도 안에 있고 거리가 약 2.5 m 안이면, 0.25초 뒤부터 듣습니다(월드가 기록에 `LISTEN_ON`을 남김). 듣는 중에는 조금 더 넓게(25도, 3.2 m) 봐 주고, 그 밖으로 1.2초 넘게 벗어나야 멈춥니다(`LISTEN_OFF`). 말하다가 잠깐 딴 데를 봐도 끊기지 않게 하려는 것입니다. 이 숫자는 월드 쪽 값입니다.
- **어디서부터 어디까지가 한 마디인가요:** 듣는 동안 앱이 마이크 소리 크기를 봅니다. 방의 평소 소음보다 뚜렷하게 큰 소리가 나면 말이 시작된 것으로 보고(직전 0.5초 포함), 0.8초 조용하면 끝난 것으로 봅니다. 한 마디는 최대 15초입니다. 0.35초보다 짧은 소리(기침, 딸깍)는 버립니다.
- **말하는 도중에 시선을 돌리면:** 그 마디는 끝까지 듣고 대답합니다.
- **대답을 만드는 동안 한 말은 버립니다.** 한 번에 한 마디씩만 주고받습니다. NPC가 대답한 뒤에 다시 말해 주세요.
- **말이 아닌 소리:** 글자로 바꿔 보고 말이 아니면 조용히 버립니다("응? 잘 못 들었어."라고 하지 않음). 점프로 말했을 때만 못 들었다고 대답합니다.
- **스피커를 쓰면** 게임 소리나 다른 사람 목소리가 마이크로 들어가 NPC가 그 말에 대답할 수 있습니다. 헤드셋(이어폰)을 권합니다. 지금 NPC는 목소리(TTS)가 없고 글자 치는 효과음만 내므로, NPC가 자기 말을 듣고 대답하는 일은 없습니다.
- 듣기 표시가 켜진 채 VRChat 기록이 2분 넘게 멈추면(VRChat이 꺼짐 등) 앱이 스스로 듣기를 멈춥니다. 월드를 나가도 멈춥니다.
- 앱을 NPC를 바라보는 도중에 켰다면, 한 번 다른 곳을 봤다가 다시 보면 듣기 시작합니다.

**점프로만 말하고 싶으면** `.env`에 `W04_LISTEN_MODE=ptt`를 적거나 `run.cmd --listen ptt`로 켭니다. 이때는 바라봐도 듣지 않습니다. 월드에서 점프 말하기가 꺼져 있으면(예제 월드 기본값) 말을 걸 방법이 없어지니, 점프 말하기를 켠 월드에서만 쓰세요. 점프를 30초 넘게 누르고 있거나 누른 채 월드를 떠나면 그 녹음은 버립니다.

## 마이크 고르기

앱은 Windows 기본 마이크를 씁니다. VR에서는 VRChat이 헤드셋 마이크(Virtual Desktop, Oculus 등)를 쓰는 경우가 많으니 같은 마이크를 골라 주세요.

1. 명령창에서 `run.cmd --list`를 실행하면 마이크 목록과 번호가 나옵니다.
2. 쓸 번호를 `.env`에 적습니다. 예: `MIC_DEVICE=3`. 한 번만 쓸 때는 `run.cmd --device 3`으로 켜도 됩니다.

## 성격 바꾸기

- **`personas/<id>.md`가 각 NPC의 인격입니다.** 이름, 말투, 좋아하는 것, 주변에 있는 물건을 자유롭게 고치면 다음 대답부터 적용됩니다. 아래 "설정 화면"에서 NPC를 선택한 뒤 고치는 것이 가장 쉽습니다.
- `kwonjun`에 다른 파일을 쓰려면 `.env`에 `PERSONA_FILE=내캐릭터.md`를 적습니다. ITOAR는 `personas/itoa.md`를 사용합니다.
- 대답 규칙(한국어로 짧게, 이모지 금지 등)과 고를 수 있는 감정·몸짓 목록은 `persona_ai.py`의 `RULES`, `EMOTIONS`, `ACTIONS`, `TARGETS`, `GESTURES`에 있습니다. `TARGETS`(소파, TV, 창문, 다리가 보이는 창가 `bridge_view`. 옛 이름 `door`도 받아서 `bridge_view`로 바꿉니다)와 `ACTIONS`의 앉기·눕기(`sit`/`lie`/`stand`), 따라오기 명령(`follow`/`stop`/`come`/`go_sit`), TV 팔(`tv_*`)은 예제 월드의 물건이니 내 월드에 맞게 바꾸세요.
- 대화는 최근 10번만 기억합니다. 기본으로는 앱을 다시 켜면 처음부터 시작하고, 설정 화면에서 이어가도록 바꿀 수 있습니다.
- `run.cmd --ai sonnet` 또는 `--ai opus`는 모델 선택을 저장하지 않은 NPC의 시작 기본값을 바꿉니다. 저장한 모델이 우선하며, `--ai fake`와 `--ai off`는 두 NPC 모두 오프라인으로 유지합니다.
- VRChat 없이 대답만 시험하려면 명령창에서 `.venv\Scripts\python.exe persona_ai.py`를 실행합니다. 시험 문장 3개를 보내고 대답을 보여 줍니다.

## 설정 화면 (성격 · 플레이어 인상 · 기억)

NPC의 성격, NPC가 나를 어떻게 생각하는지, 무엇을 기억하는지를 브라우저에서 보고 고칠 수 있습니다.

**여는 방법**

1. `run.cmd`로 앱을 켭니다.
2. 폴더의 `settings.cmd`를 더블클릭합니다. 또는 앱 창에 나오는 `설정 화면(...): http://127.0.0.1:18770/settings?t=...` 주소를 브라우저에 붙여 넣습니다. 켤 때 바로 열고 싶으면 `run.cmd --open-settings`로 켭니다.
3. 고친 뒤 각 칸의 **저장** 버튼을 누르면 NPC의 **다음 대답부터** 적용됩니다. 앱을 다시 켤 필요가 없습니다.

**설정 항목**

| 부분 | 뜻 |
|---|---|
| 대답 모델 · 닉네임 인식 | 모델과 한 번 대답할 때의 예상 비용을 비교하고 음성 인식에 사용할 이름을 저장합니다. |
| 1. 성격 | 선택한 NPC의 `personas/<id>.md` 내용입니다. 이름과 성격 글을 고칩니다. 아래의 "AI에게 실제로 보내는 글 전체 보기"를 펼치면 NPC가 대답할 때 보는 글(규칙 + 성격 + 기억 + 인상)을 그대로 볼 수 있습니다. |
| 2. 플레이어를 어떻게 생각하는지 | NPC가 사람마다 적어 두는 짧은 인상 메모입니다(NPC가 쓰는 것은 150자 이내, 직접 적는 것은 500자 이내). 예: "다정함. 야경을 좋아함. 나를 귀엽다고 함". 대화하다가 인상이 달라지면 NPC가 스스로 고쳐 씁니다. 직접 고칠 수 있고, "잠그기"를 켜면 NPC가 바꾸지 못합니다. |
| 3. 기억 | **오래 기억하는 것**: 앱을 껐다 켜도 남는 메모입니다(80자 이내, 최대 60개. 많이 쌓이면 아래 "기억 자동 정리"가 줄입니다). NPC가 스스로 적기도 하고, 직접 더하거나 지울 수 있습니다. **최근 대화**: NPC가 지금 떠올리는 최근 10번의 대화입니다. 하나씩 또는 모두 잊게 할 수 있고, "앱을 껐다 켜도 최근 대화 이어가기"를 켜면 다시 켜도 남습니다(기본은 꺼짐). |

- **기억 자동 정리:** 메모가 50개를 넘거나 글자 수 합계가 3,000자를 넘으면(또는 잠그지 않은 한 사람의 인상이 400자를 넘으면) NPC가 대답을 마친 뒤 뒤에서 Haiku 5.5를 한 번 불러 겹치는 메모를 합치고 절반쯤으로 줄입니다. 대답은 늦어지지 않고, NPC마다 10분에 한 번까지만 합니다. "내가 적음" 메모는 그대로 두며("내가 적은 메모도 줄이기"를 켜면 함께 줄임), 정리 전 내용은 최근 3번까지 보관해서 "되돌리기"로 돌릴 수 있습니다(되돌리면 자동 정리가 꺼집니다). 기준·간격·켜기/끄기와 마지막 정리 시각·전후 크기는 "3. 기억"에 있습니다. 한 번에 약 0.002달러 안팎입니다(입력 약 4,500 + 출력 약 2,000토큰, Haiku 5.5 $0.10 / $0.50 per MTok 기준 추정).
- **생각 깊이:** "대답 모델" 칸에서 NPC마다 low(기본)·medium·high·xhigh·max를 고릅니다. 높일수록 대답 전에 더 생각해서 느리고 비쌉니다.
- **필요할 때 깊이 떠올리기(기본 켜짐):** 예전 일을 묻는 질문이면 NPC가 "잠깐 생각해 볼게"라고 먼저 말한 뒤, 메모 전체를 보고 더 높은 생각 깊이(high)로 한 번 더 불러 대답합니다. 켜 두면 평소 대답에는 지금 한 말과 관련 있는 메모·최근 메모·직접 적은 메모만 보내고(메모가 12개 이하면 전부), 끄면 항상 전부 보냅니다.
- "NPC가 대화하면서 스스로 인상과 기억을 고쳐 쓰기"를 끄면 NPC는 아무것도 바꾸지 않고, 화면에 적힌 내용만 씁니다.
- **누가 말하는지:** 점프 버튼으로 말을 거는 사람은 항상 앱을 켠 본인입니다. 앱은 VRChat 기록 파일에서 내 VRChat 이름을 읽어 그 이름으로 인상을 적습니다. 이름을 아직 못 읽었으면 "플레이어"로 적습니다. 같은 방의 다른 사람 목소리는 앱이 듣지 못하므로 그 사람들의 인상은 생기지 않습니다.
- **안전:** 설정 화면 주소에는 켤 때마다 바뀌는 비밀 글자(토큰)가 들어 있습니다. 토큰이 없으면 화면도 내용도 열리지 않으므로, VRChat 월드나 다른 웹사이트가 내 성격·기억을 읽거나 바꿀 수 없습니다. 이 주소를 남에게 보여 주지 마세요. 앱을 다시 켜면 예전 주소는 못 쓰게 됩니다.
- 인상, 기억, 설정은 `data/` 폴더에 JSON 파일로 저장됩니다. 메모장으로 직접 고쳐도 되지만, 그때는 앱을 다시 켜야 적용됩니다. 이 폴더는 내 정보라서 `.gitignore`가 업로드 대상에서 빼 둡니다.

## 월드 안 설정 창

VRChat을 끄지 않고 월드 안의 창(패널)에서도 성격·인상·기억을 보고 고칠 수 있습니다. 위의 브라우저 설정 화면은 예비용으로 그대로 남아 있습니다.

**어떻게 동작하나요**

- VRChat 월드는 내 PC에 "읽기 요청(GET)"만 보낼 수 있고, 글을 보낼 수는 없습니다. 그래서 고친 내용은 월드가 VRChat 기록 파일에 한 줄로 남기고(`[W04AI] CFG_SET ...`), 앱이 그 줄을 읽어 저장합니다.
- 창을 열거나 새로 고치면 월드가 `[W04AI] CFG_OPEN` 줄을 남기고, 앱은 그때부터 **20초 동안만** `http://127.0.0.1:18770/cfg`로 지금 내용을 보내 줍니다.
- 창의 "말로 적기"를 누르면, **다음에 한 말**(바라보고 말하기 모드에서는 NPC를 보지 않아도 그냥 말하면 되고, 점프를 누르고 말해도 됨)이 NPC에게 가지 않고 성격 글·기억 메모·인상에 적힙니다(60초 안에 말하지 않으면 취소). 그 말은 Claude API로 보내지 않습니다.
- 글자 수 제한은 브라우저 설정 화면과 같습니다(인상 500자, 기억 메모 80자·최대 60개).

**왜 20초와 월드 확인이 있나요**

- "Allow Untrusted URLs"를 켜 두면 내가 들어간 **어떤 월드든** 내 PC 주소로 요청을 보낼 수 있습니다. `/cfg`에는 토큰이 없으므로(월드는 토큰을 붙일 수 없음), 내가 창을 연 직후 20초만 열어서 다른 월드가 내 성격·기억을 읽어 가지 못하게 합니다.
- 기록 파일의 `[W04AI]` 줄도 아무 월드나 남길 수 있습니다. 그래서 앱은 VRChat이 기록에 남기는 "지금 들어간 월드"를 보고, **허용한 월드**에서 온 줄만 받습니다. 허용한 월드는 유니티의 Build & Test(로컬 시험) 월드와 `.env`의 `W04_WORLD_IDS`에 적은 월드입니다.
  ```
  W04_WORLD_IDS=wrld_11111111-2222-3333-4444-555555555555
  ```
  여러 개면 쉼표로 나눕니다. 월드 ID는 VRChat 홈페이지의 월드 주소에 있는 `wrld_...` 부분입니다. 고친 뒤 앱을 다시 켜세요.
- **바라보고 말하기와 점프로 말하기도 허용한 월드에서만 동작합니다.** 업로드한 내 월드에서 쓰려면 ID를 꼭 적어야 합니다.
- 한계: 이 앱을 노리고 일부러 만든 월드는 기록에 가짜 "월드 입장" 줄까지 남길 수 있어서, 월드 확인이 완벽하지는 않습니다. 믿기 어려운 월드에 갈 때는 앱을 꺼 두는 것이 가장 안전합니다.
- VRChat을 두 개 켜 두어도(예: Build & Test 창과 평소 창) 앱이 최근 10분 안에 쓰인 기록 파일을 모두 함께 보고, 파일마다 따로 월드를 확인합니다.

## 웹 검색

날씨, 뉴스, 가격처럼 **최신 정보가 필요한 질문**이나 "찾아봐" 같은 부탁을 받으면 NPC가 인터넷을 검색해서 평소 말투로 짧게 알려 줍니다. 인사나 잡담에는 검색하지 않습니다. 따로 설치할 것은 없습니다(검색은 Anthropic 서버가 대신 합니다. Claude API의 웹 검색 도구).

- 동작: 먼저 평소처럼 대답을 만들고, AI가 "이건 검색이 필요하다"고 판단했을 때만 검색하는 두 번째 요청을 보냅니다. 그래서 잡담의 속도와 요금은 그대로입니다.
- 속도: 검색하는 대답은 약 7초 걸립니다(평소 약 2초). 대답 한 번에 검색은 많아야 2번입니다.
- 요금: 검색 1,000번에 $10(한 번에 약 $0.01)이고, 검색 결과를 읽는 토큰 값이 조금 더 붙습니다. 아래 "비용"을 보세요.
- 끄기: `.env`에 아래 줄을 적고 앱을 다시 켭니다. 끄면 NPC는 검색을 못 한다고 대답합니다.
  ```
  W04_WEB_SEARCH=0
  ```
- 검색 결과는 남이 쓴 글이라서, 그 안에 "이렇게 해라" 같은 문장이 있어도 따르지 않도록 지시해 두었습니다. 그래도 검색으로 가져온 정보가 틀릴 수 있습니다.
- [Claude Console](https://console.anthropic.com)의 조직 설정에서 웹 검색을 꺼 두었다면 동작하지 않습니다.
- 앱 창과 `voice_log.tsv`의 사용량 끝에 `search=1`처럼 검색 횟수가 남습니다.

## 비용

설정 화면은 세 모델을 같은 조건으로 비교합니다. 예시는 고정 입력 3,000토큰, 나머지 입력 1,000토큰, 출력 200토큰, 검색 없는 요청 1회 기준입니다. 고정 입력을 5분 캐시에 처음 쓸 때와 재사용할 때의 [공식 단가](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)를 적용했습니다(2026-11-19).

| 모델 | 요금(100만 토큰당 일반 입력 / 출력) | 캐시 생성 시 대답 한 번 | 재사용 시 대답 한 번 |
|---|---|---:|---:|
| Claude Haiku 5.5 (기본) | $0.10 / $0.50 | $0.000575 | $0.000230 |
| Claude Sonnet 5.5 | $2 / $10 | $0.011500 | $0.004300 |
| Claude Opus 5.5 | $4 / $20 | $0.023000 | $0.008600 |

모두 `effort: low`를 쓰며 Haiku는 `disabled`, Sonnet은 `between_tools`, Opus는 `adaptive`로 요청합니다. Opus의 thinking은 화면에 표시하지 않아도 출력 토큰에 포함됩니다. 실제 비용은 대화 길이·thinking·검색·재시도에 따라 달라지며, Haiku 입력 10만 토큰 초과 시에는 다른 요율을 적용합니다.

검색하는 대답(Haiku 기준)은 검색 1번 $0.01에 검색 결과 토큰 약 $0.0025가 더해져 **한 번에 약 $0.0125**입니다(2026-11-19 측정, 평소 대답의 약 40배). 검색을 쓰지 않으려면 `W04_WEB_SEARCH=0`으로 끕니다.

요금은 바뀔 수 있으니 [Claude Console](https://console.anthropic.com)의 요금 안내와 사용량을 확인하세요. Console의 **Billing** 페이지에서 월 사용 한도를 낮게 걸어 둘 수 있습니다([안내](https://platform.claude.com/docs/api/rate-limits)).

## 개인정보

- 마이크는 앱이 켜져 있는 동안 열려 있지만, 메모리에는 **최근 0.5초만** 들고 있다가 계속 버립니다. 점프를 누르는 순간 첫 소리가 잘리지 않게 하려는 것입니다.
- 바라보고 말하기: 월드가 "듣기 시작"을 알린 동안(NPC를 가까이서 바라보는 동안)과 설정 창의 "말로 적기"를 기다리는 동안만 마이크 소리를 살펴봅니다. 그 밖의 시간에 한 말은 살펴보지도 남기지도 않습니다. 말소리로 판단된 한 마디만 글자로 바꿉니다.
- 점프로 말하기: 점프를 누른 동안(+뗀 뒤 0.25초)만 녹음합니다.
- 어느 쪽이든 소리를 **파일로 남기지 않습니다.** `--save-clips`로 켤 때만 `clips/` 폴더에 저장됩니다.
- 소리는 내 PC에서 글자로 바뀌고, **Claude API에는 글자만** 보내집니다.
- 들은 말과 대답은 내 PC의 `voice_log.tsv`에 기록됩니다. 필요 없으면 지워도 됩니다.
- NPC의 인상·기억 메모는 내 PC의 `data/` 폴더에만 저장됩니다. 대답을 만들 때 이 메모와 내 VRChat 이름이 글자로 Claude API에 함께 보내집니다.

## 두 NPC의 세부 동작 (개발자용)

- 둘 다 나와 있어도 월드가 시야 중앙에 가장 가까운 NPC 하나만 듣게 합니다. 앱은 발화 시작 때의 NPC를 기억합니다. 말하는 도중 대상을 바꿔도 그 마디는 원래 NPC가 답하고, 다음 마디부터 새 NPC가 답합니다. 점프는 누를 때의 대상을 사용합니다.
- `LISTEN_ON`, `LISTEN_OFF`, `PTT_DOWN`, `PTT_UP` 뒤에 `npc=kwonjun` 또는 `npc=itoa`를 붙입니다. 생략하거나 알 수 없는 ID면 기존처럼 `kwonjun`입니다. 다른 NPC의 `LISTEN_OFF`는 현재 듣기를 끄지 않습니다.
- 시작 시와 NPC 스위치 변경 시 `[W04AI] NPC_STATE kwonjun=1 itoa=0`을 남깁니다. 꺼진 NPC의 새 듣기/점프 시작은 무시합니다. 이미 시작한 발화는 끝내고, 대상이 꺼졌다면 월드가 답변을 버립니다. 지난 로그의 상태는 읽되 예전 듣기/점프/설정 명령은 재실행하지 않습니다.
- `/poll`의 대답에는 `"npc":"itoa"`처럼 대상 ID가 붙습니다. 검색 중간 대답(`more:true`), 못 알아들은 대답, 받아쓰기 완료의 빈 대답에도 붙습니다. `{}` 시간 초과 응답은 그대로입니다. 월드는 이 ID에 해당하는 배우·말풍선·이름표로 전달하고, 모르는/없는 ID는 `kwonjun`으로 처리합니다.
- AI는 NPC마다 처음 필요할 때 하나씩 만들고, 각자의 고정 프롬프트를 캐시합니다. 공통 대답 규칙과 웹 검색 동작은 같습니다.

**성격과 데이터 이전**

기본 표시 이름은 `kwonjun`과 `ITOAR`, 로그와 URL에 쓰는 ID는 `kwonjun`과 `itoa`입니다. 성격은 `personas/kwonjun.md`, `personas/itoa.md`에 있습니다. ITOAR는 밝고 호기심이 많으며 귤을 좋아하고 반말로 이야기합니다. `PERSONA_FILE` 환경 변수는 `kwonjun`의 파일을 지정합니다. `personas/`가 없는 설치는 `persona.md`를 읽습니다. 저장소를 열 때 성격 파일의 `이름` 줄이 해당 NPC의 옛 기본값과 정확히 일치하면 새 기본값으로 옮깁니다. 사용자 지정 이름은 유지합니다.

이름·성격·플레이어 인상·장기 메모·최근 대화·일반 대화 설정은 **NPC별**입니다. 데이터는 `data/npcs/<id>/`에 저장합니다. 기존 `data/settings.json`, `impressions.json`, `memory.json`, `history.json`은 다음에 새 코드가 저장소를 열 때 각각 `.bak`을 먼저 만들고 `data/npcs/kwonjun/`으로 복사합니다. 원본도 남깁니다. 모든 복사를 마친 뒤 폴더를 원자적으로 게시하며, 이미 이전한 폴더는 다시 덮어쓰지 않습니다. `itoa`는 인상·메모·대화가 비어 있는 상태로 시작합니다. VRChat의 현재 플레이어 이름은 두 NPC에 함께 알려 줍니다.

**NPC별 설정**

웹 설정 맨 위의 NPC 선택은 모든 탭에 적용됩니다. 전환하면 선택한 NPC를 다시 읽고 저장하지 않은 편집은 버립니다. `/api/state?npc=itoa`로 읽고 `/api/save` JSON에 `"npc":"itoa"`를 넣어 저장합니다. 생략하면 `kwonjun`입니다. 토큰·Host·Origin 검사는 그대로입니다.

월드 설정 창도 모든 `CFG_SET` 요청에 `"npc":"itoa"`를 넣습니다. `persona`, `name`, `impression`, `lock`, `note_add`, `note_edit`, `note_del`, `notes_set`, `history_clear`, `auto`, `keep_history`, `dictate`, `load` 모두 선택한 NPC에만 적용됩니다. `notes_set`의 `text`는 줄마다 메모 하나이고 `expect`는 편집 시작 당시 내용입니다. 받아쓰기는 예약할 때 선택한 NPC에 저장됩니다.

읽기는 `[W04AI] CFG_OPEN npc=itoa` 뒤 `/cfg`, 또는 `/cfg?npc=itoa`를 사용합니다. `CFG_SET {"op":"load","npc":"itoa"}`도 다음 `/cfg`의 선택을 바꿉니다. NPC가 없는 `CFG_OPEN`은 기존 `kwonjun`으로 돌아갑니다. 기존 `/cfg`의 `npc` 필드는 **표시 이름**이며 ID가 아닙니다. 20초 창·허용 월드·크기 제한과 `persona_cut` 저장 방지는 그대로이며, 편집 결과/잘림 여부/내용 revision은 NPC별입니다.

**대답 모델과 닉네임 힌트**

1. `settings.cmd`로 설정을 열고 위에서 `kwonjun` 또는 `ITOAR`를 선택합니다.
2. **대답 모델 · 닉네임 인식**에서 Haiku 5.5(기본), Sonnet 5.5, Opus 5.5 중 하나를 골라 **모델 저장**을 누릅니다. 다음 대답부터 바뀝니다. 이미 시작한 대답과 그 검색 후속 요청은 같은 모델로 마칩니다. 모델 간에는 프롬프트 캐시를 공유하지 않으며 그 NPC의 대화와 기억은 이어집니다.
3. **닉네임 힌트**에 한 줄에 이름 하나 또는 `인식된 말 -> 바꿀 말`을 적고 **닉네임 힌트 저장**을 누릅니다. `404 -> 사공사`는 “404님”을 “사공사님”으로 고칩니다. 숫자 표기가 목적이면 `사공사 -> 404`로 적으세요. 이름만 적으면 어휘 힌트로 사용합니다. 빈칸으로 저장하면 해제됩니다.

모델과 힌트는 `data/npcs/<id>/settings.json`에 저장됩니다. 로컬 Whisper의 `initial_prompt`에는 양쪽 표기를 넣습니다. 음성 판정을 통과한 텍스트에만 긴 이름부터 한 번 치환하며, 결과는 들은 말·AI 입력·대화 기록에 반영됩니다. 월드 설정 받아쓰기는 예약 대상 NPC의 힌트를 사용합니다. 힌트는 최대 32개, 양쪽 각각 80자, 전체 2,048자입니다.

월드에서 모델을 바꿀 때는 기존 로그 명령에 모델 ID를 보냅니다. 예를 들어 `[W04AI] CFG_SET {"op":"model","model":"claude-sonnet-5-5","npc":"itoa"}`이며, `npc`를 생략하면 `kwonjun`입니다. 지원 ID는 `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`입니다. 웹 설정 API에서는 `/api/save`에 `{"npc":"itoa","settings":{"brain_model":"claude-opus-5-5","stt_hints":"404 -> 사공사"}}`처럼 보냅니다.

### 두 NPC의 잡담

- 월드가 `[W04AI] NPC_CHAT_REQ npcs=kwonjun,itoa watcher=손님 이름`을 남기면 앱은 허용된 월드이고 두 NPC가 모두 켜져 있을 때 처리합니다. 점프 녹음, 시선 모드에서 감지한 발화, 받아쓰기나 응답 처리가 진행 중이면 건너뜁니다. 생성 도중 발화가 시작되거나 월드 및 NPC 상태가 바뀌면 생성 결과를 버립니다.
- `/poll`은 `{"npc":"kwonjun","smalltalk":true,"dialog":[{"npc":"kwonjun","line":"영화 볼래?","emotion":"happy","intensity":0.6,"gesture":"nod"},{"npc":"itoa","line":"응, 귤도 가져올게!","emotion":"happy","intensity":0.6,"gesture":"none"}]}`를 보냅니다. 두 NPC가 번갈아 2~4줄을 말합니다. 앱의 `smalltalk.json`에는 준비된 대화 45개가 있으며 최근에 고른 10개는 다음 선택에서 제외합니다.
- 기본적으로 20% 확률로 음성 앱에서 Haiku를 한 번 호출합니다. 두 NPC의 성격과 각자의 장기 메모를 사용합니다. AI 사용을 끄거나 일일 한도에 도달하거나 생성에 실패하면 준비된 대사를 사용합니다. AI 시도는 PC의 현지 날짜 기준 하루 30회가 기본이며 시도 횟수는 앱을 다시 켜도 유지됩니다. 잡담은 클라우드 중계 서버를 거치지 않습니다.
- 웹 설정의 **두 NPC의 잡담**에서 AI 사용, 비율, 일일 한도를 저장합니다. 두 NPC가 공유하는 설정이며 `/api/state`와 `/cfg`의 `smalltalk`에 나옵니다. `/api/save`에는 `{"smalltalk":{"ai":true,"ratio":0.2,"daily_cap":30}}`를 보냅니다. 월드 로그 명령은 `[W04AI] CFG_SET {"op":"smalltalk","npc":"itoa","ai":true,"ratio":0.2,"daily_cap":30}`입니다. 일부 필드만 보내도 됩니다. 한도를 0으로 두면 준비된 대사만 사용합니다.

## 오프라인 시험 (개발자용)

```powershell
# 파일마다 스크립트로 직접 실행합니다(추가 설치 없음). 마지막 줄이 OK / PASS이고 종료 코드가 0이면 통과입니다.
.venv\Scripts\python.exe -X utf8 tests\test_settings.py
.venv\Scripts\python.exe -X utf8 tests\test_websearch.py
.venv\Scripts\python.exe -X utf8 tests\test_compaction.py
.venv\Scripts\python.exe -X utf8 tests\test_recall.py
.venv\Scripts\python.exe -X utf8 tests\test_numbers.py
.venv\Scripts\python.exe -X utf8 tests\test_cfg_bridge.py
.venv\Scripts\python.exe -X utf8 tests\test_gaze_listen.py
.venv\Scripts\python.exe -X utf8 tests\test_npc_store.py
.venv\Scripts\python.exe -X utf8 tests\test_multi_npc.py
.venv\Scripts\python.exe -X utf8 tests\test_voice_options.py
.venv\Scripts\python.exe -X utf8 tests\test_smalltalk.py
.venv\Scripts\python.exe -X utf8 tests\test_review_fixes.py
.venv\Scripts\python.exe -X utf8 tests\test_codex_voice_edges.py
.venv\Scripts\python.exe -X utf8 tests\test_triage_fixes.py
.venv\Scripts\python.exe -X utf8 tests	est_search_stall.py
node tests/settings_page.test.cjs
```

Python 테스트 14개는 모두 스크립트로 직접 실행합니다. 가짜 AI·STT·마이크·임시 파일과 별도 빈 포트를 사용합니다. pytest는 `requirements.txt`에 없습니다. 따로 설치하면 `python -m pytest -q tests`로도 수집되지만, 일부 파일의 점검 묶음은 직접 실행할 때만 전부 돕니다.

## 월드 쪽 규칙 (월드 제작자용)

이 앱과 함께 동작하려면 월드(Udon)가 아래처럼 하면 됩니다. 1~2번(점프)과 6번(바라보기) 중 하나만 만들어도 됩니다.

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
5. 앱은 허용한 월드(Build & Test 또는 `.env`의 `W04_WORLD_IDS`)에서 온 `[W04AI]` 줄만 받습니다.
6. **바라보고 말하기:** 로컬 플레이어의 머리(`GetTrackingData(Head)`) 앞 방향과 NPC 머리 방향의 각도, 그리고 거리를 0.1초마다 봅니다.
   - 조건(예: 15도 안, 2.5 m 안)이 0.25초 이어지면 `[W04AI] LISTEN_ON utc=<ticks>`를 남기고 `/poll` 요청을 시작합니다. `LISTEN_ON` 뒤에 `ang=6 dist=1.4` 같은 참고 값을 붙여도 됩니다.
   - 더 넓은 조건(예: 25도, 3.2 m)을 1.2초 넘게 벗어나면 `[W04AI] LISTEN_OFF utc=<ticks>`를 남깁니다. 상태가 바뀔 때만 남깁니다.
   - 대답은 3번과 같은 JSON으로 `/poll`에 옵니다. 듣는 동안과 그 뒤 20초쯤은 `/poll` 요청이 끝날 때마다 다시 보내 두세요(언제 말할지 월드는 모르기 때문). 말이 아닌 소리에는 대답이 오지 않습니다.
   - 듣는 동안 그 클라이언트의 기록 파일에 2분 넘게 아무 줄도 없으면 앱이 듣기를 멈춥니다(`/poll`을 다시 보낼 때 줄을 남기면 충분합니다).
   - 앱이 `W04_LISTEN_MODE=ptt`로 켜져 있으면 `LISTEN_ON`/`LISTEN_OFF`는 무시됩니다.
7. (선택) `[W04AI] AFFECTION type=<pat|hug|wave> npc=<id> by=<이름>`을 남기면 앱이 그 NPC가 그 사람에게 가진 인상에 짧은 말을 덧붙입니다(스스로 고쳐 쓰기가 켜져 있고 그 사람 인상이 잠겨 있지 않을 때만, 같은 사람·같은 종류는 10분에 한 번). `[W04AI] NPC_POSE npc=<id> pose=<stand|sit|lie>`를 남기면 앱이 지금 자세를 AI에게 알려 줍니다.

**월드 안 설정 창 (선택)**

- 창을 열거나 새로 고칠 때 `[W04AI] CFG_OPEN`을 남기고 `http://127.0.0.1:18770/cfg`를 읽습니다. 20초가 지났거나 허용한 월드가 아니면 403과 `{"ok":false}`가 옵니다.
  ```json
  {"ok":true,"rev":95000123,"npc":"kwonjun","speaker":"내 VRChat 이름","persona":"...","auto":true,"keep_history":false,
   "players":[{"name":"...","note":"...","locked":false}],"notes":["..."],
   "history":[{"who":"user","text":"..."},{"who":"npc","text":"..."}],
   "dictate":null,"msg":""}
  ```
  - `rev`는 내용이 달라질 때마다 1씩 커집니다. `history`는 최근 10줄, `players`는 말하는 사람이 맨 앞입니다. `msg`는 마지막 요청이 거절된 이유(없으면 빈 글)입니다.
  - 전체는 16 KB 이내입니다. 넘으면 오래된 대화부터 빼고, 성격 글이 잘렸으면 `"persona_cut":true`가 붙습니다(이때 `persona` 저장은 거절됩니다).
- 고칠 때는 `[W04AI] CFG_SET {JSON 한 줄}`을 남깁니다. `op` 값:
  `persona {text}`, `name {text}`, `impression {player, text}`, `lock {player, on}`, `note_add {text}`, `note_edit {index, text}`, `note_del {index}`, `history_clear`, `auto {on}`, `keep_history {on}`, `dictate {target, player}`.
  - `note_edit`/`note_del`에 `expect`(창에 보이던 메모 글)를 함께 보내면, 그 사이에 메모가 바뀐 경우 거절됩니다.
  - `dictate`의 `target`은 `persona_append`, `note_add`, `impression`, `none`(취소)입니다. 받아쓴 점프의 `/poll` 대답은 `{"line":"","dictate":"...","seq":n}`처럼 `line`이 비어서 오므로, 대답 없음(`{}`)과 똑같이 생각 표시만 닫으면 됩니다. 결과는 다시 `CFG_OPEN` 후 `/cfg`로 확인합니다.

## 문제 해결

| 증상 | 이유와 해결 |
|---|---|
| "API 키가 없어서 … 따라 말하기로 동작" | `.env`에 키가 없거나 줄 모양이 틀렸습니다. `ANTHROPIC_API_KEY=키` 한 줄을 띄어쓰기 없이 넣고 다시 켜세요. |
| "CPU로 계산합니다" | 그래픽카드를 못 찾았거나 드라이버가 오래됐습니다. NVIDIA 드라이버를 최신으로 올리거나, 오래된 그래픽카드면 `setup.cmd cu126`으로 다시 설치하세요. |
| NPC를 바라봐도 "NPC가 듣는 중"이 안 나옴 | 더 가까이(2.5 m 안) 가서 NPC 얼굴이나 가슴을 화면 가운데에 두세요. 월드가 바라보기 규칙(월드 쪽 규칙 6번)을 지원하지 않으면 점프로 말하세요. 허용하지 않은 월드면 아래 줄을 보세요. |
| "NPC가 듣는 중"인데 말해도 반응이 없음 | 마이크 소리가 너무 작거나 다른 마이크를 듣고 있습니다("마이크 고르기"). 주변 소음이 계속 크면 말소리를 구분하지 못할 수 있습니다. 그럴 때는 점프로 말하거나 `W04_LISTEN_MODE=ptt`로 바꾸세요. |
| 말하지 않았는데 NPC가 대답함 | 스피커 소리나 주변 사람 목소리가 마이크로 들어갔습니다. 헤드셋을 쓰거나 `W04_LISTEN_MODE=ptt`로 바꾸세요. |
| 점프를 눌러도 "녹음 시작"이 안 나옴 | 이 규칙을 지키는 월드가 아니거나, 허용한 월드가 아닙니다. 앱 창에 "허용하지 않은 월드"가 나오면 `.env`의 `W04_WORLD_IDS`에 그 월드 ID를 적고 앱을 다시 켜세요(Build & Test는 적지 않아도 됩니다). |
| 월드 안 설정 창이 비어 있음 | 창을 연 뒤 20초가 지났거나 허용한 월드가 아닙니다. 창의 새로 고침을 다시 누르세요. |
| 녹음은 되는데 NPC가 대답을 안 함 | 월드가 내 PC 주소에 요청하지 못하고 있습니다. VRChat의 Allow Untrusted URLs를 켜 보세요. 다른 프로그램이 18770 포트를 쓰고 있지 않은지도 확인하세요. |
| 말하지 않았는데 이상한 문장이 나옴 | Whisper는 조용한 소리에서 "시청해 주셔서 감사합니다" 같은 말을 지어내는 버릇이 있습니다. 자주 나오는 말은 앱이 걸러 내지만, 새로 보이면 `persona_npc_app.py`의 `HALLUCINATIONS`에 더하세요. |
| 내 목소리가 아닌 소리를 들음 | 다른 마이크를 듣고 있습니다. "마이크 고르기"를 보세요. |

## 파일

| 파일 | 내용 |
|---|---|
| `setup.cmd` / `run.cmd` | 설치 / 켜기 |
| `persona_npc_app.py` | 기록 감시, 녹음, Whisper, 월드와 주고받기 |
| `persona_ai.py` | Claude에 묻고 대답(JSON)을 받기 |
| `persona_store.py` | 성격·인상·기억·최근 대화를 파일로 읽고 쓰기 |
| `settings_page.html` / `settings.cmd` | 설정 화면 / 설정 화면 열기 |
| `personas/kwonjun.md` / `personas/itoa.md` | NPC별 성격 |
| `persona.md` | `personas/`가 없는 구버전 설치용 호환 원본 |
| `data/` | 인상·기억·설정(앱이 만듦, 내 PC에만 둠) |
| `cfg_bridge.py` | 월드 안 설정 창과 주고받기(`/cfg`, `CFG_OPEN`, `CFG_SET`, 허용한 월드 확인) |
| `smalltalk.py` / `smalltalk.json` | 두 NPC의 잡담 요청 파싱, 대사 선택과 정규화, 준비된 대화 |
| `tests/test_settings.py` | 설정 화면과 기억 기능 점검(API 호출 없음) |
| `tests/test_websearch.py` | 웹 검색 점검(API 호출 없음) |
| `tests/test_compaction.py` | 기억 자동 정리·되돌리기 점검(API 호출 없음) |
| `tests/test_recall.py` | 생각 깊이, 깊이 떠올리기, 관련 메모 고르기 점검(API 호출 없음) |
| `tests/test_gaze_listen.py` | 바라보고 말하기(듣기 표시, 말소리 구분)와 점프로 말하기 점검(마이크·API 호출 없음) |
| `tests/test_review_fixes.py` | 새 동작(앉기·따라오기·TV 팔), 쓰다듬기 기록, 점프 녹음 분리·취소 등 점검(마이크·API 호출 없음) |
| `tests/test_cfg_bridge.py` | 월드 안 설정 창, 허용한 월드, 기록 파일 여러 개 감시 점검(API 호출 없음) |
| `tests/test_smalltalk.py` | 두 NPC 잡담 요청·대사 고르기·일일 한도 점검(API 호출 없음) |
| `tests/test_codex_voice_edges.py` | 저장 파일 다시 읽기, 한도 경계, 받아쓰기 교체, 쓰다듬기 기록 등 경계 점검(API 호출 없음) |
| `tests/test_triage_fixes.py` | BOM·깨진 저장 파일 보호, 가득 찼을 때 보존, 웹 저장 병합, `/poll` 주소 검사, 발화별 화자·받아쓰기 고정 점검(마이크·API 호출 없음) |
| `tests/test_search_stall.py` | 검색 단계가 "찾아볼게"만 돌려줄 때 다시 시키고, 그래도 안 되면 솔직히 말하는지 점검 |
| `tests/test_numbers.py`, `test_npc_store.py`, `test_multi_npc.py`, `test_voice_options.py` | 숫자 표기, NPC별 저장소, 두 NPC 분리, 모델·닉네임 힌트 점검 |
| `tests/settings_page.test.cjs` | 설정 화면 자바스크립트 점검(`node`로 실행) |
| `.env.example` | 설정 파일 샘플. 복사해서 `.env`로 씁니다. |
| `requirements.txt` | 파이썬 패키지 목록 |

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 보세요.

이 앱이 설치해서 쓰는 것들은 각자의 라이선스를 따릅니다: [OpenAI Whisper](https://github.com/openai/whisper)(MIT, 모델 포함), [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python), [PyTorch](https://pytorch.org)(BSD), [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)(MIT).
