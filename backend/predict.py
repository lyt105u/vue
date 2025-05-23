# usage:
# python predict.py model.json file --data_path input.xlsx --output_name result
# python predict.py model.json input --input_values 1.0 TRUE 3.0 FALSE
# read_csv 和 read_excel 會自行轉換數值型 (int 或 float)和布林型 (bool)

# import subprocess
# import sys
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "scikit-learn==1.5.2"],
#     stdout=subprocess.DEVNULL
# )
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "--upgrade", "xgboost"],
#     stdout=subprocess.DEVNULL
# )

import argparse
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os
import json
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import sys

def load_model(model_path):
    if model_path.lower().endswith(".json"):  # .json 結尾
        model = XGBClassifier()
        model.load_model(model_path)
        # print(f"XGBoost 模型已從 {model_path} 載入")
    elif model_path.lower().endswith(".pkl"):
        model = joblib.load(model_path)
        # print(f"模型已從 {model_path} 載入")
    elif model_path.lower().endswith(".zip"):
        model = TabNetClassifier()
        model.load_model(model_path)
        # print(f"模型已從 {model_path} 載入")
    else:
        print(json.dumps({
            "status": "error",
            "message": f"Unsupported model format: {model_path}",
        }))
        sys.exit(1)
    return model

def load_data(data_path):
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path)
    # print(f"Data loaded from {data_path}")
    return data

def predict_labels(model, model_path, data, label_column):
    x_test = data.copy()
    if model_path.lower().endswith(".zip"): # tabnet 只吃 numpy array，不吃 object
        x_test = x_test.to_numpy(dtype=np.float32)
    else:
        x_test = x_test.values
    y_pred = model.predict(x_test)
    data[label_column] = y_pred
    return data

def save_predictions(data, data_path, output_name):
    result_dir = os.path.join("data", "result")
    os.makedirs(result_dir, exist_ok=True)

    # output_path = os.path.join(result_dir, output_name)
    if data_path.endswith(".csv"):
        output_path = os.path.join(result_dir, f"{output_name}.csv")
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif data_path.endswith(".xlsx"):
        output_path = os.path.join(result_dir, f"{output_name}.xlsx")
        data.to_excel(output_path, index=False)
    # print(f"Predictions saved to {output_path}")
    print(json.dumps({
        "status": "success",
        "message": f"Predictions saved to {output_path}",
    }))

def predict_input(model, model_path, input_values):
    # 將輸入值解析為適當的型態
    parsed_values = []
    for value in input_values:
        if isinstance(value, str):
            if value.upper() == "TRUE":
                parsed_values.append(1)
            elif value.upper() == "FALSE":
                parsed_values.append(0)
            else:
                try:
                    parsed_values.append(float(value))
                except ValueError:
                    raise ValueError(f"無法解析輸入值：{value}，請確認格式正確！")
        else:
            parsed_values.append(float(value))

    # 預測結果
    if model_path.lower().endswith(".zip"): # tabnet 只吃 numpy array，不吃 object
        prediction = model.predict(np.array([parsed_values], dtype=np.float32))
    else:
        prediction = model.predict([parsed_values])

    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    print(json.dumps({
        "status": "success",
        "message": prediction,
    }))

def main(model_path, mode, data_path=None, output_name=None, input_values=None, label_column="label"):
    try:
        model = load_model(model_path)

        if mode == "file":
            data = load_data(data_path)
            data_with_predictions = predict_labels(model, model_path, data, label_column)
            save_predictions(data_with_predictions, data_path, output_name)
        elif mode == "input":
            predict_input(model, model_path, input_values)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the trained model file (e.g., .json for XGBoost or .pkl for joblib models)")
    parser.add_argument('mode', type=str, choices=["file", "input"], help="模式選擇：file（檔案模式）或 input（手動輸入模式）")
    parser.add_argument('--data_path', type=str, help="輸入檔案路徑（僅 file 模式需要）")
    parser.add_argument('--output_name', type=str, help="輸出檔案名稱（僅 file 模式需要）")
    parser.add_argument('--input_values', type=str, nargs='+', help="輸入特徵數值（僅 input 模式需要，數值間以空格分隔）")
    parser.add_argument('--label_column', type=str, help="自訂標籤欄位名稱")
    
    args = parser.parse_args()
    main(args.model_path, args.mode, args.data_path, args.output_name, args.input_values, args.label_column)
