@echo off
rem VRC AI Persona NPC - open the settings page of the running app in the browser.
chcp 65001 >nul
cd /d "%~dp0"
if not exist "data\settings_url.txt" (
  echo 먼저 run.cmd 로 앱을 켜 주세요. / Start the app with run.cmd first.
  pause
  exit /b 1
)
set /p W04_URL=<"data\settings_url.txt"
start "" "%W04_URL%"
