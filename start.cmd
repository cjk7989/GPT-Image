@echo off
cd /d "%~dp0"
cd src
echo Starting GPT Image Generator...
echo http://localhost:8000
uvicorn main:app --reload --port 8000
pause
