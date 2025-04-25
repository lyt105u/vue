# usage: python deleteTabRows.py data.csv "[1,3,5]"
import pandas as pd
import os
import argparse
import ast
import json

def delete_rows(file_path, row_indices):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        print(json.dumps({
            "status": "error",
            "message": "Only .csv or .xlsx files are supported.",
        }))

    df = df.drop(index=row_indices, errors='ignore').reset_index(drop=True)

    if ext == ".csv":
        df.to_csv(file_path, index=False)
    else:
        df.to_excel(file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete specific rows from CSV or XLSX and overwrite the file.")
    parser.add_argument("file", help="Path to the CSV or XLSX file.")
    parser.add_argument("rows", help="Python-style list of row indices to delete, e.g. [1,3,5]")

    args = parser.parse_args()
    file_path = args.file
    try:
        row_indices = ast.literal_eval(args.rows)
        if not isinstance(row_indices, list) or not all(isinstance(i, int) for i in row_indices):
            print(json.dumps({
                "status": "error",
                "message": "Row indices must be a list of integers.",
            }))
        delete_rows(file_path, row_indices)
        print(json.dumps({
            "status": "success",
            "message": f"Deleted rows {row_indices} from {file_path}",
        }))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        
