import sys
import json
import pandas as pd
import numpy as np

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

def apply_fill_strategy(df: pd.DataFrame, missing_methods: dict):
    try:
        for column, method in missing_methods.items():
            if column not in df.columns:
                continue
            if method == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif method == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif method == 'mode':
                mode_series = df[column].mode()
                if not mode_series.empty:
                    df[column].fillna(mode_series[0], inplace=True)
            elif method == 'max':
                df[column].fillna(df[column].max(), inplace=True)
            elif method == 'min':
                df[column].fillna(df[column].min(), inplace=True)
            elif method == 'zero':
                df[column].fillna(0, inplace=True)
            elif method == 'skip':
                continue
            else:
                print(json.dumps({
                    "status": "error",
                    "message": f"Unknown method: {method}",
                }))
                sys.exit(1)
        return df
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Failed to apply fill strategy: {e}",
        }))
        sys.exit(1)

def save_file(df: pd.DataFrame, file_path: str):
    try:
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        else:
            raise ValueError("Unsupported file format for saving.")
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Failed to save file: {e}",
        }))
        sys.exit(1)

def main():
    # 讀取參數
    file_path = sys.argv[1]
    missing_methods = json.loads(sys.argv[2])

    # 讀取資料
    df = load_file(file_path)
    # 將空欄位改成 np.nan，才能被 fillna 抓出來
    df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}, inplace=True)

    # 補值邏輯
    apply_fill_strategy(df, missing_methods)  # 不回傳
    save_file(df, file_path)

    print(json.dumps({
        "status": "success",
        "message": f"Filled. Remaining missing: {df.isna().sum().to_dict()}",
        "file_path": file_path
    }))

if __name__ == '__main__':
    main()
