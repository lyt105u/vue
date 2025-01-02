import os
import sys

def list_file_names(param):
    if param == "data":
        folder_path = "data/"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"folder {folder_path} does not exist")
        # 列出所有 .xlsx 檔案
        return [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    elif param == "model":
        folder_path = "model/"
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"folder {folder_path} does not exist")
        # 列出所有檔案
        return [file for file in os.listdir(folder_path)]

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("Invalid number of arguments. Expected exactly one argument.")

        param = sys.argv[1]

        file_names = list_file_names(param)
        for file in file_names:
            print(file)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)
