@echo off
rem VRC AI Persona NPC - one-time setup. Double-click, or: setup.cmd [cu130|cu126|cpu]
rem   cu130 (default with an NVIDIA GPU): RTX 50/40/30/20 series, needs a recent driver (580+)
rem   cu126: older NVIDIA GPUs (GTX 10 series) or older drivers / cpu: no NVIDIA GPU
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY python --version >nul 2>&1 && set PY=python
if not defined PY (
  echo [오류] 파이썬을 찾지 못했습니다. https://www.python.org 에서 Python 3.10 이상을 설치해 주세요.
  echo [error] Python not found. Install Python 3.10+ from https://www.python.org
  goto :fail
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || (
  echo [오류] Python 3.10 이상이 필요합니다. / Python 3.10+ is required.
  goto :fail
)

set TORCH=%1
if "%TORCH%"=="" (
  where nvidia-smi >nul 2>&1 && (set TORCH=cu130) || (set TORCH=cpu)
)

echo [1/4] 가상환경 만들기 (.venv) / creating .venv
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto :fail
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip >nul

echo [2/4] PyTorch 설치 (%TORCH%, 2~3 GB일 수 있음) / installing PyTorch (%TORCH%)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/%TORCH% || goto :fail

echo [3/4] 나머지 패키지 설치 / installing requirements
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :fail

echo [4/4] 설정 파일 준비 / preparing .env
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo .env 파일을 만들었습니다. 메모장 등으로 열어 ANTHROPIC_API_KEY= 뒤에 키를 넣어 주세요.
) else (
  echo .env 파일이 이미 있어서 그대로 둡니다.
)
".venv\Scripts\python.exe" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음 (CPU로 동작)')"
echo.
echo 설치 완료. run.cmd 를 더블클릭해서 켜세요. / Setup done. Double-click run.cmd.
pause
exit /b 0

:fail
echo.
echo 설치에 실패했습니다. 위의 오류 메시지를 확인해 주세요. / Setup failed, see the messages above.
pause
exit /b 1
