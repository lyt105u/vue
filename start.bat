@echo off
start cmd /k "cd /d frontend\ && npm run serve"
start cmd /k "cd /d backend\ && myenv\Scripts\activate && python app.py"

REM 等待後端啟動 (假設 Flask 運行在 http://127.0.0.1:5000)
:WAIT_BACKEND
timeout /t 5 /nobreak >nul
curl -s http://127.0.0.1:5000 >nul 2>nul
if errorlevel 1 goto WAIT_BACKEND

REM 等待前端啟動 (假設前端在 http://localhost:8080)
:WAIT_FRONTEND
timeout /t 5 /nobreak >nul
curl -s http://localhost:8080 >nul 2>nul
if errorlevel 1 goto WAIT_FRONTEND

REM 兩個都啟動後，打開瀏覽器
start "" http://localhost:8080
