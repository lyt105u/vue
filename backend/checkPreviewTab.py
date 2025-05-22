# 檢查表格資料是否缺漏，並提供預覽資訊
# usage: python checkPreviewTab.py <file_path>
# ex:
#   python checkPreviewTab.py upload/tabular.csv

import os
import sys
import json
import pandas as pd

def load_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            print(json.dumps({
                "status": "error",
                "message": "Support .csv or .xlsx files only.",
            }))
            sys.exit(1)
        return df
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        sys.exit(1)

def number_to_excel_col(n):
    col = ""
    while n >= 0:
        col = chr(n % 26 + ord('A')) + col
        n = n // 26 - 1
    return col

def check_missing(df):
    coords = list(zip(*df.isnull().to_numpy().nonzero()))
    missing_positions = []

    for row_idx, col_idx in coords:
        col_letter = number_to_excel_col(col_idx)
        excel_row = row_idx + 2  # 一律 +2，因為假設 row 0 是資料第一列，row 1 是 Excel 第3列
        missing_positions.append(f"{col_letter}{excel_row}")

    return missing_positions

def preview(df, max_rows=10, max_columns=30):
    total_rows = len(df)
    total_columns = len(df.columns)
    preview_df = df.head(max_rows)

    return {
        "columns": df.columns.tolist(),
        "preview": preview_df.fillna('').to_dict(orient='records'),
        "total_rows": total_rows,
        "total_columns": total_columns
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: `python checkPreviewTab.py <file_path>`.",
        }))
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_file(file_path)

    if df is not None:
        missing_coords = check_missing(df)
        if missing_coords:
            print(json.dumps({
                "status": "errorMissing",
                "message": f"Missing data: {missing_coords}"
            }))
        else:
            preview_data = preview(df)
            print(json.dumps({
                "status": "success",
                "preview_data": preview_data
            }))
