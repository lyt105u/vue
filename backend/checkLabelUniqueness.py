# usage: python checkLabelUniqueness.py "upload/<username>/川崎症病人(訓練用 第一階段).csv" label

import pandas as pd
import argparse
import sys
import os
import json

def check_label_uniqueness(file_path, label_column):
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(json.dumps({
            "status": "error",
            "message": "<file_path> doesn't exist.",
        }))
        sys.exit(1)

    # 根據副檔名判斷讀取方式
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print(json.dumps({
                "status": "error",
                "message": f"Unsupported file type: {file_path}.",
            }))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Failed to load file: {e}.",
        }))
        sys.exit(1)

    # 檢查欄位是否存在
    if label_column not in df.columns:
        print(json.dumps({
            "status": "error",
            "message": f"Label column '{label_column}' not found in file.",
        }))
        sys.exit(1)

    # 檢查唯一分類數量
    unique_classes = df[label_column].nunique()
    if unique_classes < 2:
        print(json.dumps({
            "status": "errorUnique",
            "message": f"Only one class '{df[label_column].unique()[0]}' found in label column '{label_column}'. Cannot train a model.",
        }))
    else:
        print(json.dumps({
            "status": "success",
            "message": f"Found {unique_classes} unique classes in '{label_column}': {df[label_column].unique()}.",
        }))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if label column has more than one class.")
    parser.add_argument("file_path", help="Path to the CSV or Excel file.")
    parser.add_argument("label_column", help="Name of the label column to check.")
    args = parser.parse_args()

    check_label_uniqueness(args.file_path, args.label_column)
