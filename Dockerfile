# -- Build Vue 前端 --
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/ .
RUN npm install && npm run build

# -- 建立 Flask + gunicorn --
FROM python:3.11-slim
WORKDIR /app

# 安裝 Python 套件
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 Flask 檔案與 Vue 靜態檔
COPY backend/ .
COPY --from=frontend-builder /app/dist ./static

# 若有版本檢查腳本
RUN python tool_checkVersion.py

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
