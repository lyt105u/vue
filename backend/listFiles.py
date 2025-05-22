# 列出 folder 資料夾內的檔案
# 若沒給副檔名，則全列
# 副檔名可以給不只一個
# usage: python listFiles.py <folder> <副檔名>
# ex:
#   python listFiles.py upload          -> 全列
#   python listFiles.py upload txt      -> 列出 txt
#   python listFiles.py upload csv xlsx -> 列出 csv 和 xlsx

import os
import sys
import json
import argparse

def list_files_by_extensions(folder_path, extensions):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Normalize extensions to ensure they start with a dot
    normalized_exts = [ext if ext.startswith(".") else "." + ext for ext in extensions]

    matched_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # If no extension filtering or file matches one of the extensions
            if not extensions or any(file.lower().endswith(ext) for ext in extensions):
                matched_files.append(file)  # Only return file name, not full path
    return matched_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "message": "Missing arguments. Usage: python listFiles.py <folder> <extension>",
        }))
        sys.exit(1)

    parser = argparse.ArgumentParser(description="List files with specific extensions (or all files if no extension is provided).")
    parser.add_argument("folder", help="The folder path to list files from.")
    parser.add_argument("extensions", nargs='*', help="File extensions to filter by (e.g. txt csv). Dot prefix is optional.")

    args = parser.parse_args()
    folder_path = args.folder
    extensions = args.extensions

    try:
        files = list_files_by_extensions(folder_path, extensions)
        print(json.dumps({
            "status": "success",
            "files": files,
        }))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": e,
        }))
