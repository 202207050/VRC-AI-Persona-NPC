@echo off
rem VRC AI Persona NPC - check the Claude API key (never shows the whole key)
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 먼저 setup.cmd 를 실행해 주세요. / Run setup.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 check_key.py
pause
