@echo off
rem W04 voice app - double-click to start. Close this window to stop.
rem Precision transcription (beam 5): about +0.1 s on RTX 5080, fewer misheard words (2026-11-20).
set W04_WHISPER_BEAM_SIZE=5
call "%~dp0run.cmd" %*
