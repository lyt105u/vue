import subprocess
import os

# ===== 可調整參數 =====
BASE_DIR = "/home/m123040053"
IMAGE_NAME = "m123040053_mlas"
PORT_OUT = 8081
PORT_IN = 5000
# =====================

# 自動組合
CONTAINER_NAME = f"{IMAGE_NAME}_container"
HOST_UPLOAD_DIR = os.path.join(BASE_DIR, "vue/backend/upload")
HOST_MODEL_DIR = os.path.join(BASE_DIR, "vue/backend/model")
HOST_WHITE_LIST = os.path.join(BASE_DIR, "vue/backend/listWhite.txt")

def run_cmd(cmd: str):
    print(f"執行中: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"指令失敗: {cmd}")
    else:
        print(f"完成: {cmd}")

def main():
    # 停用舊容器（若已存在）
    run_cmd(f"docker stop {CONTAINER_NAME}")

    # 刪除舊容器（若已存在）
    run_cmd(f"docker rm {CONTAINER_NAME}")

    # 刪除舊 image
    run_cmd(f"docker rmi {IMAGE_NAME}")

    # 建立 Docker image（正常建置 & no-cache 建置）
    run_cmd(f"docker build -t {IMAGE_NAME}:latest .")

    # 啟動新容器
    run_cmd(
        f"docker run --gpus all -d "
        f"-p {PORT_OUT}:{PORT_IN} "
        f"--name {CONTAINER_NAME} "
        f"-v {HOST_UPLOAD_DIR}:/app/upload "
        f"-v {HOST_MODEL_DIR}:/app/model "
        f"-v {HOST_WHITE_LIST}:/app/listWhite.txt "
        f"{IMAGE_NAME}:latest"
    )

if __name__ == "__main__":
    main()
