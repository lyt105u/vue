# usage:
# python predict.py model.json file --file_path input.xlsx --output_name result.xlsx
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

def load_model(model_path):
    filepath = os.path.join("model", model_path)
    if filepath.lower().endswith(".json"):  # .json 結尾
        model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                gamma=0, gpu_id=-1, importance_type=None,
                interaction_constraints='', learning_rate=0.300000012,
                max_delta_step=0, max_depth=6, min_child_weight=1,
                monotone_constraints='()', n_estimators=100, n_jobs=72,
                num_parallel_tree=1, predictor='auto', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                tree_method='exact', validate_parameters=1, verbosity=None)
        model.load_model(filepath)
        # print(f"XGBoost 模型已從 {filepath} 載入")
    elif filepath.lower().endswith(".pkl"):
        model = joblib.load(filepath)
        # print(f"模型已從 {filepath} 載入")
    else:
        raise ValueError(f"無法識別的模型格式或檔案：{filepath}")
    return model

def load_data(data_path):
    full_path = os.path.join("data", "predict", data_path)
    if full_path.endswith(".csv"):
        data = pd.read_csv(full_path)
    elif full_path.endswith(".xlsx"):
        data = pd.read_excel(full_path)
    # print(f"Data loaded from {full_path}")
    return data

def predict_labels(model, data):
    x_test = data.values
    y_pred = model.predict(x_test)
    data['label'] = y_pred
    # print("Predictions added to the dataset")
    return data

def save_predictions(data, data_path, output_name):
    result_dir = os.path.join("data", "result")
    os.makedirs(result_dir, exist_ok=True)

    # output_path = os.path.join(result_dir, output_name)
    if data_path.endswith(".csv"):
        output_path = os.path.join(result_dir, f"{output_name}.csv")
        data.to_csv(output_path, index=False, encoding='utf-8')
    elif data_path.endswith(".xlsx"):
        output_path = os.path.join(result_dir, f"{output_name}.xlsx")
        data.to_excel(output_path, index=False)
    # print(f"Predictions saved to {output_path}")
    print(json.dumps({
        "status": "success",
        "message": f"Predictions saved to {output_path}",
    }))

def predict_input(model, input_values):
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
    prediction = model.predict([parsed_values])
    if isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    print(json.dumps({
        "status": "success",
        "message": prediction,
    }))

def main(model_path, mode, data_path=None, output_name=None, input_values=None):
    model = load_model(model_path)

    if mode == "file":
        data = load_data(data_path)
        data_with_predictions = predict_labels(model, data)
        save_predictions(data_with_predictions, data_path, output_name)
    elif mode == "input":
        if not input_values:
            raise ValueError("在輸入模式下，必須提供特徵數值作為參數！")
        predict_input(model, input_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the trained model file (e.g., .json for XGBoost or .pkl for joblib models)")
    parser.add_argument('mode', type=str, choices=["file", "input"], help="模式選擇：file（檔案模式）或 input（手動輸入模式）")
    parser.add_argument('--data_path', type=str, help="輸入檔案路徑（僅 file 模式需要）")
    parser.add_argument('--output_name', type=str, help="輸出檔案名稱（僅 file 模式需要）")
    parser.add_argument('--input_values', type=str, nargs='+', help="輸入特徵數值（僅 input 模式需要，數值間以空格分隔）")
    
    args = parser.parse_args()
    main(args.model_path, args.mode, args.data_path, args.output_name, args.input_values)
