# 檢查表格資料是否缺漏，並提供預覽資訊
# usage: python checkPreviewTab.py <file_path>
# ex:
#   python checkPreviewTab.py upload/<username>/tabular.csv

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
    missing_col_indices = set()

    for row_idx, col_idx in coords:
        # 缺失座標
        col_letter = number_to_excel_col(col_idx)
        excel_row = row_idx + 2  # 一律 +2，因為假設 row 0 是資料第一列，row 1 是 Excel 第3列
        missing_positions.append(f"{col_letter}{excel_row}")
        missing_col_indices.add(col_idx)
    # 根據 df.columns 的原始順序選出有缺失的欄位名稱
    headers = [col_name for idx, col_name in enumerate(df.columns) if idx in missing_col_indices]
    
    return missing_positions, headers

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

def get_summary(df):
    summary_data = {}
    numeric_df = df.select_dtypes(include='number')  # 僅處理數值欄位
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        mode_values = col_data.mode()
        mode = float(mode_values[0]) if not mode_values.empty else None  # 只取第一個眾數
        mean_val = col_data.mean() if not col_data.empty else None
        std_val = col_data.std() if not col_data.empty else None
        summary_data[col] = {
            "mean": float(mean_val) if mean_val is not None else None,
            "median": float(col_data.median()) if not col_data.empty else None,
            "min": float(col_data.min()) if not col_data.empty else None,
            "max": float(col_data.max()) if not col_data.empty else None,
            "std": float(std_val) if std_val is not None else None,
            "mode": mode,
            "zscore_min": float(((col_data - mean_val) / std_val).min()) if std_val not in (None, 0) else None,
            "zscore_max": float(((col_data - mean_val) / std_val).max()) if std_val not in (None, 0) else None,
        }
    return summary_data

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
        missing_coords, missing_headers = check_missing(df)
        summary_data = get_summary(df)
        preview_data = preview(df)
        print(json.dumps({
            "status": "success",
            "preview_data": preview_data,
            "summary": summary_data,
            "missing_cords": missing_coords,
            "missing_header": missing_headers
        }))
