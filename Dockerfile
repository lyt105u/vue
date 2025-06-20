# -- Build Vue 前端 --
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/ .
RUN npm install && npm run build

# -- 建立 Flask + gunicorn --
FROM python:3.11-slim
WORKDIR /app

# 安裝必要系統依賴，給 C/C++ 編譯用
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-dev \
    libssl-dev \
    libffi-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製 Flask 檔案與 Vue 靜態檔
COPY backend/ .
COPY --from=frontend-builder /app/dist ./static

# 若有版本檢查腳本
RUN python tool_checkVersion.py

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
