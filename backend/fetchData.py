# 列出 data\ 內的 xlsx 和 csv，或列出 model\ 內的所有檔案
# usage: python fetchData.py <data 或 model>

import os
import sys
import json

def list_file_names(param):
    if param == "data/train":
        folder_path = "data/train/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        files = [file for file in os.listdir(folder_path) if file.endswith(('.xlsx', '.csv'))]
        print(json.dumps({
            "status": "success",
            "files": files
        }, ensure_ascii=False))

    elif param == "data/predict":
        folder_path = "data/predict/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        files = [file for file in os.listdir(folder_path) if file.endswith(('.xlsx', '.csv'))]
        print(json.dumps({
            "status": "success",
            "files": files
        }, ensure_ascii=False))

    elif param == "model":
        folder_path = "model/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        files = [file for file in os.listdir(folder_path) if file != "field_counts.json"]
        print(json.dumps({
            "status": "success",
            "files": files
        }))
    
    else:
        print(json.dumps({
            "status": "error",
            "message": "Invalid parameter. Use 'data' or 'model'.",
            "files": []
        }))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Missing or incorrect arguments.",
            "files": []
        }))
        sys.exit(1)

    param = sys.argv[1]
    list_file_names(param)
