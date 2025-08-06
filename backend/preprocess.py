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

def apply_rules(df: pd.DataFrame, rules_json):
    modified_columns = set()
    rows_affected = set()

    for rule in rules_json:
        col = rule.get("column")
        expect_type = rule.get("expect_type")
        fallback_type = rule.get("fallback_type")

        if col not in df.columns:
            continue

        # 找出不符合的列
        if expect_type == "not_missing":
            condition = df[col].isna()
        elif expect_type == "condition":
            cond = rule.get("expect_condition")
            val = rule.get("expect_value")
            try:
                val = float(val)
            except:
                continue

            if cond == ">=":
                condition = ~(df[col] >= val)
            elif cond == "<=":
                condition = ~(df[col] <= val)
            elif cond == "!=":
                condition = ~(df[col] != val)
            else:
                continue
        else:
            continue

        # 處理 fallback
        if fallback_type == "drop":
            drop_indices = condition[condition].index
            df.drop(index=drop_indices, inplace=True)
            rows_affected.update(drop_indices.tolist())
        elif fallback_type == "skip":
            continue
        else:
            if fallback_type == "custom":
                fallback_value = rule.get("fallback_value")
            elif fallback_type == "min":
                fallback_value = df[col].min()
            elif fallback_type == "max":
                fallback_value = df[col].max()
            elif fallback_type == "mean":
                fallback_value = df[col].mean()
            elif fallback_type == "median":
                fallback_value = df[col].median()
            elif fallback_type == "mode":
                fallback_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
            else:
                continue

            if fallback_value is not None:
                affected_rows = condition[condition].index.tolist()
                if affected_rows:
                    df.loc[condition, col] = fallback_value
                    modified_columns.add(col)
                    rows_affected.update(affected_rows)

    return list(modified_columns), len(rows_affected)

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
    rules_json = json.loads(sys.argv[2])

    # 讀取資料
    df = load_file(file_path)
    # 將空欄位改成 np.nan，才能被 fillna 抓出來
    df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}, inplace=True)

    # 補值邏輯
    modified_columns, affected_count = apply_rules(df, rules_json)
    save_file(df, file_path)

    print(json.dumps({
        "status": "success",
        "modified_columns": modified_columns,
        "rows_affected": affected_count,
        "message": f"Modified {len(modified_columns)} columns across {affected_count} rows.",
        "file_path": file_path
    }, ensure_ascii=False))

if __name__ == '__main__':
    main()
