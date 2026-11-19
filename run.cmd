@echo off
rem VRC AI Persona NPC - start the companion app. Close this window to stop.
rem   run.cmd --list         list microphones
rem   run.cmd --device 3     use microphone #3 (or set MIC_DEVICE in .env)
rem   run.cmd --ai sonnet    answer with Claude Sonnet instead of Haiku
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 먼저 setup.cmd 를 실행해 주세요. / Run setup.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -u persona_npc_app.py %*
pause
