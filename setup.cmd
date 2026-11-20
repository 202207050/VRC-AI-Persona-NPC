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
  echo [안내] 파이썬이 없습니다. / Python not found.
  where winget >nul 2>&1 || (
    echo https://www.python.org 에서 Python 3.12를 설치한 뒤 setup.cmd 를 다시 실행해 주세요.
    echo Install Python 3.12 from https://www.python.org and run setup.cmd again.
    goto :fail
  )
  choice /c YN /m "지금 Python 3.12를 자동으로 설치할까요? / Install Python 3.12 now"
  if errorlevel 2 goto :fail
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements || goto :fail
  echo.
  echo 파이썬 설치가 끝났습니다. 이 창을 닫고 setup.cmd 를 한 번 더 더블클릭해 주세요.
  echo Python installed. Close this window and double-click setup.cmd again.
  pause
  exit /b 0
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
if not exist ".env" copy ".env.example" ".env" >nul
findstr /r /c:"^ANTHROPIC_API_KEY=..*" ".env" >nul 2>&1 && (
  echo API 키가 이미 들어 있어서 그대로 둡니다. / API key already set.
) || call :askkey
".venv\Scripts\python.exe" -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음 (CPU로 동작)')"

echo.
choice /c YN /m "바탕화면에 'VRC AI NPC' 바로가기를 만들까요? / Create a desktop shortcut"
if not errorlevel 2 powershell -NoProfile -Command "$d=[Environment]::GetFolderPath('Desktop'); $l=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'VRC AI NPC.lnk')); $l.TargetPath='%~dp0run.cmd'; $l.WorkingDirectory='%~dp0'; $l.IconLocation='%SystemRoot%\System32\imageres.dll,180'; $l.Save()" && echo 바탕화면에 'VRC AI NPC' 를 만들었습니다.

echo.
echo 설치 완료. 다음부터는 바탕화면의 'VRC AI NPC' 또는 run.cmd 로 켜세요.
echo Setup done. Next time start it with 'VRC AI NPC' on the desktop or run.cmd.
choice /c YN /m "지금 바로 켤까요? / Start now"
if errorlevel 2 exit /b 0
endlocal
call "%~dp0run.cmd"
exit /b 0

:askkey
echo.
echo Claude API 키가 필요합니다. https://console.anthropic.com 에서 만든 키(sk-ant-로 시작)를
echo 복사해서 여기에 붙여 넣고 Enter를 누르세요. (마우스 오른쪽 클릭 = 붙여넣기)
echo 키는 이 PC의 .env 파일에만 저장됩니다. 지금 없으면 그냥 Enter (나중에 .env에 넣어도 됨).
echo Paste your Claude API key (sk-ant-...) and press Enter. Empty = skip.
set "W04_KEY="
set /p "W04_KEY=API 키 / API key: "
if not defined W04_KEY (
  echo 건너뜀. 키 없이 켜면 들은 말을 따라 하는 시험 모드로 동작합니다.
  exit /b 0
)
powershell -NoProfile -Command "$p=Join-Path (Get-Location) '.env'; $k=$env:W04_KEY.Trim().Trim([char]34); $l=[IO.File]::ReadAllLines($p) | ForEach-Object { if ($_ -match '^ANTHROPIC_API_KEY=') { 'ANTHROPIC_API_KEY=' + $k } else { $_ } }; [IO.File]::WriteAllLines($p, [string[]]$l, (New-Object Text.UTF8Encoding $false))" || (
  echo [오류] 키를 저장하지 못했습니다. .env 파일을 메모장으로 열어 직접 넣어 주세요.
  exit /b 0
)
set "W04_KEY="
echo API 키를 .env 에 저장했습니다. / Saved to .env.
exit /b 0

:fail
echo.
echo 설치에 실패했습니다. 위의 오류 메시지를 확인해 주세요. / Setup failed, see the messages above.
pause
exit /b 1
