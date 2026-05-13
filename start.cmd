@echo off
cd /d "%~dp0"
cd src
echo Starting GPT Image Generator...
echo http://127.0.0.1:8000
start "" http://127.0.0.1:8000
uvicorn main:app --reload --port 8000
pause
