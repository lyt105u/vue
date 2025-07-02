# 抓出使用者資料夾內，每個 task_dir 內 status.json 的內容
# usage:
#   python listStatus.py model/alice

import os
import sys
import json
import argparse

def collect_user_status(user_dir):
    result = []

    os.makedirs(user_dir, exist_ok=True)
    # 遍歷每個 task_id 子資料夾
    for task_id in os.listdir(user_dir):
        task_dir = os.path.join(user_dir, task_id)
        if not os.path.isdir(task_dir):
            continue

        status_path = os.path.join(task_dir, "status.json")
        if not os.path.exists(status_path):
            continue

        try:
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 加上 task_id
            data["task_id"] = task_id
            result.append(data)

        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Failed to read {status_path}: {str(e)}"
            }))
            sys.exit(1)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "Missing folder argument. Usage: python listStatus.py <user_folder>"
        }))
        sys.exit(1)

    parser = argparse.ArgumentParser(description="List all status.json under a user's task folders.")
    parser.add_argument("folder", help="The user folder path (e.g. model/alice)")
    args = parser.parse_args()

    try:
        status_list = collect_user_status(args.folder)
        print(json.dumps({
            "status": "success",
            "data": status_list
        }, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }))
        sys.exit(1)
