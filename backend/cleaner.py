import os
import time
import threading
import shutil

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪檔案或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 遞迴刪資料夾
        except Exception as e:
            print(f"[Cleaner] Failed to delete {file_path}: {e}")
    print(f"[Cleaner] Folder {file_path} cleaned successfully!")

def scheduled_clean():
    while True:
        now = time.localtime()
        if now.tm_hour == 19 and now.tm_min == 00:  # 每天 00:00 (在 docker 上是 UTC+0)
            print("[Cleaner] Cleaning folders...")
            clean_folder("upload")
            clean_folder("model")
            time.sleep(60)  # 等 1 分鐘，避免多次清理
        else:
            time.sleep(30)
