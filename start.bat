@echo off
start cmd /k "cd /d frontend\ && npm run serve"
start cmd /k "cd /d backend\ && myenv\Scripts\activate && python app.py"
