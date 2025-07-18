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
from tool_train import NumpyEncoder, extract_base64_images_and_clean_json
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os
import json
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import sys
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
try:
    # 嘗試使用 Docker 中的 NotoSansCJK 字型
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()

except Exception:
    # Fallback：改用本機字型清單
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']

# 顯示負號正常
matplotlib.rcParams['axes.unicode_minus'] = False
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

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

def explain_with_shap(model, x_test, feature_names):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        is_tabnet = (
            hasattr(model, "predict_proba") and
            not hasattr(model, "predict")  # TabNet 沒有原生 predict，只有 predict_proba
        ) or model.__class__.__name__.lower().startswith("tabnet")

        if is_tabnet:
            # 專門為 TabNet 使用 KernelExplainer
            background = x_test[:20]
            explain_data = x_test[:5]

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(explain_data)
            shap_data = explain_data

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values_for_plot = shap_values[:, :, 1]
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_for_plot = shap_values[1]
            else:
                shap_values_for_plot = shap_values

            if shap_values_for_plot.ndim != 2:
                raise ValueError(f"shap_values shape is invalid: {shap_values_for_plot.shape}")
        
        elif isinstance(model, Pipeline) and "sgdclassifier" in model.named_steps:
            estimator = model.named_steps["sgdclassifier"]
            scaler = model.named_steps["standardscaler"]
            x_scaled = scaler.transform(x_test)

            explainer = shap.LinearExplainer(estimator, x_scaled, feature_perturbation="interventional")
            shap_values = explainer.shap_values(x_scaled)
            shap_data = x_scaled

            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[1]
            else:
                shap_values_for_plot = shap_values

        elif isinstance(model, MLPClassifier):
            # 使用 lambda 包裝 MLPClassifier，避免 KernelExplainer 報錯
            background = x_test[:20]
            explain_data = x_test[:5]

            predict_fn = lambda x: model.predict_proba(x)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(explain_data)
            shap_data = explain_data

            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values_for_plot = shap_values[:, :, 1]
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_for_plot = shap_values[1]
            else:
                shap_values_for_plot = shap_values

            if shap_values_for_plot.ndim != 2:
                raise ValueError(f"shap_values shape is invalid: {shap_values_for_plot.shape}")

        else:
            # 其他模型使用 shap.Explainer（TreeExplainer, etc.）
            explainer = shap.Explainer(model, x_test)
            shap_values = explainer(x_test, check_additivity=False)
            shap_data = shap_values.data

            if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                shap_values_for_plot = shap_values.values[..., 1]
            else:
                shap_values_for_plot = shap_values.values

        # 平均重要度
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
        }

        # beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, shap_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

    except Exception as e:
        result["shap_error"] = str(e)

    return result

def explain_with_lime(model, x_test, y_test, feature_names):
    result = {}
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=["class_0", "class_1"],
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test[0],  # 第0筆樣本
            model.predict_proba,
            num_features=10
            # num_features=len(x_test[0])
        )

        # 儲存圖片到記憶體 buffer
        fig = lime_result.as_pyplot_figure()
        fig.set_size_inches(10, 6)  # 改變圖的尺寸
        fig.tight_layout()  # 自動調整 layout
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)  # 關閉圖避免重疊

        # 同時保留文字格式
        result["lime_example_0"] = lime_result.as_list()

    except Exception as e:
        result["lime_error"] = str(e)
        
    return result

def predict_labels(model, model_path, data, label_column):
    feature_names = data.columns.tolist()
    x_test = data.copy()
    if model_path.lower().endswith(".zip"): # tabnet 只吃 numpy array，不吃 object
        x_test = x_test.to_numpy(dtype=np.float32)
    else:
        x_test = x_test.values
    y_pred = model.predict(x_test)
    data[label_column] = y_pred

    explain_result = {}
    try:
        explain_result["shap"] = explain_with_shap(model, x_test, feature_names)
        explain_result["lime"] = explain_with_lime(model, x_test, y_pred, feature_names)
    except Exception as e:
        explain_result["explain_error"] = str(e)

    return data, explain_result

def save_predictions(data, data_path, output_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)

    # output_path = os.path.join(result_dir, output_name)
    if data_path.endswith(".csv"):
        output_path = os.path.join(task_dir, f"{output_name}.csv")
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif data_path.endswith(".xlsx"):
        output_path = os.path.join(task_dir, f"{output_name}.xlsx")
        data.to_excel(output_path, index=False)
    return output_path

def predict_input(model, model_path, input_values, task_dir):
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

    result = {
        "status": "success",
        "prediction": prediction,
        "task_dir": task_dir
    }
    result_json_path = os.path.join(task_dir, "metrics.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
    extract_base64_images_and_clean_json(task_dir, "metrics.json")
    print(json.dumps(result))

def main(model_path, mode, task_dir, data_path=None, output_name=None, input_values=None, label_column="label"):
    try:
        model = load_model(model_path)

        if mode == "file":
            data = load_data(data_path)
            data_with_predictions, explanations = predict_labels(model, model_path, data, label_column)
            output_path = save_predictions(data_with_predictions, data_path, output_name, task_dir)
            result = {
                "status": "success",
                # "message": f"Predictions saved to {output_path}",
            }
            shap_result = explanations.get("shap", {})
            if shap_result.get("shap_plot"):
                result["shap_plot"] = shap_result["shap_plot"]
            if shap_result.get("shap_importance"):
                result["shap_importance"] = shap_result["shap_importance"]
            lime_result = explanations.get("lime", {})
            if lime_result.get("lime_plot"):
                result["lime_plot"] = lime_result["lime_plot"]
            if lime_result.get("lime_example_0"):
                result["lime_example_0"] = lime_result["lime_example_0"]
            result["task_dir"] = task_dir
            result_json_path = os.path.join(task_dir, "metrics.json")
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
            extract_base64_images_and_clean_json(task_dir, "metrics.json")
            print(json.dumps(result))
            
        elif mode == "input":
            predict_input(model, model_path, input_values, task_dir)
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
    parser.add_argument("task_dir", type=str)
    parser.add_argument('--data_path', type=str, help="輸入檔案路徑（僅 file 模式需要）")
    parser.add_argument('--output_name', type=str, help="輸出檔案名稱（僅 file 模式需要）")
    parser.add_argument('--input_values', type=str, nargs='+', help="輸入特徵數值（僅 input 模式需要，數值間以空格分隔）")
    parser.add_argument('--label_column', type=str, help="自訂標籤欄位名稱")
    
    args = parser.parse_args()
    main(args.model_path, args.mode, args.task_dir, args.data_path, args.output_name, args.input_values, args.label_column)
