# usage:
# python predict.py model.json file --data_path input.xlsx --output_name result
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import metrics
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
import io
import base64
from tool_train import NumpyEncoder, extract_base64_images_and_clean_json
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import zipfile
import tempfile
import shutil
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

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
    else:
        print(json.dumps({
            "status": "error",
            "message": f"Unsupported file format.",
        }))
        sys.exit(1)
    return data

def predict_labels(model, model_path, data, label_column, pred_column):
    x_test = data.drop(columns=[label_column], errors='ignore').copy()
    y_test = data[label_column].copy()
    if model_path.lower().endswith(".zip"): # tabnet 只吃 numpy array，不吃 object
        x_test = x_test.to_numpy(dtype=np.float32)
    else:
        x_test = x_test.values
    y_pred = model.predict(x_test)
    data[pred_column] = y_pred
    return x_test, y_test, y_pred, data

def save_predictions(data, data_path, output_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)

    if data_path.endswith(".csv"):
        output_path = os.path.join(task_dir, f"{output_name}.csv")
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif data_path.endswith(".xlsx"):
        output_path = os.path.join(task_dir, f"{output_name}.xlsx")
        data.to_excel(output_path, index=False)

def evaluate_model(y_test, y_pred, model, x_test):
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result = {
        "status": "success",
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred) * 100),
            "recall": float(recall_score(y_test, y_pred) * 100),
            "precision": float(precision_score(y_test, y_pred) * 100),
            "f1_score": float(f1_score(y_test, y_pred) * 100),
        }
    }

    # recall 分析
    y_pred_proba = model.predict_proba(x_test)
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []
    for th in range(1,101):
        th *= 0.01
        y_pred = [0 if (x[0] >= th) else 1 for x in y_pred_proba]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn+fp) if (tn + fp) > 0 else 0
        npv = tn / (fn+tn) if (fn + tn) > 0 else 0
        thresh_list.append(th)
        accuracy_score_list.append(accuracy_score(y_test, y_pred) * 100)
        precision_score_list.append(precision_score(y_test, y_pred, zero_division=0) * 100)
        recall_score_list.append(recall_score(y_test, y_pred, zero_division=0) * 100)
        f1_score_list.append(f1_score(y_test, y_pred, zero_division=0) * 100)
        specificity_score_list.append(specificity * 100)
        npv_score_list.append(npv * 100)
        confusion_matrix_list.append([tn, fp, fn, tp])

    recall_standard_list = [80, 85, 90, 95]
    for recall_standard in recall_standard_list:
        high_recall_f1score = []
        high_recall_accuracy = []
        high_recall_recall = []
        high_recall_precision = []
        high_recall_specificity = []
        high_recall_npv = []
        high_recall_confusion_matrix = []
        high_recall_f2score = []
        for index, recall in enumerate(recall_score_list):
            if recall >= recall_standard:
                precision = precision_score_list[index]
                high_recall_f1score.append(f1_score_list[index])
                high_recall_accuracy.append(accuracy_score_list[index])
                high_recall_precision.append(precision_score_list[index])
                high_recall_recall.append(recall_score_list[index])
                high_recall_specificity.append(specificity_score_list[index])
                high_recall_npv.append(npv_score_list[index])
                high_recall_confusion_matrix.append(confusion_matrix_list[index])
                f2_score = (5 * precision * recall) / (4 * precision + recall)
                high_recall_f2score.append(f2_score)
        
        if high_recall_f1score:
            # 檢查是否有符合的 Recall 值
            high_recall_best_f1_score_index = np.argmax(high_recall_f1score)
            best_recall = high_recall_recall[high_recall_best_f1_score_index]
            best_specificity = high_recall_specificity[high_recall_best_f1_score_index]
            best_precision = high_recall_precision[high_recall_best_f1_score_index]
            best_npv = high_recall_npv[high_recall_best_f1_score_index]
            best_f1 = high_recall_f1score[high_recall_best_f1_score_index]
            best_f2 = high_recall_f2score[high_recall_best_f1_score_index]
            best_accuracy = high_recall_accuracy[high_recall_best_f1_score_index]
            best_confusion_matrix = high_recall_confusion_matrix[high_recall_best_f1_score_index]
        else:
            # 如果沒有符合 Recall ≥ 80, 85, 90, 95，填入 0
            best_recall = 0
            best_specificity = 0
            best_precision = 0
            best_npv = 0
            best_f1 = 0
            best_f2 = 0
            best_accuracy = 0
            best_confusion_matrix = [0, 0, 0, 0]

        key_name = f"recall_{recall_standard}"
        result[key_name] = {
            "recall": best_recall,
            "specificity": best_specificity,
            "precision": best_precision,
            "npv": best_npv,
            "f1_score": best_f1,
            "f2_score": best_f2,
            "accuracy": best_accuracy,
            "true_negative": best_confusion_matrix[0],
            "false_positive": best_confusion_matrix[1],
            "false_negative": best_confusion_matrix[2],
            "true_positive": best_confusion_matrix[3]
        }

    # plot
    y_pred_roc = [x[1] for x in y_pred_proba]
    guess_tp = np.arange(0, 1, 0.001)
    guess_fp = np.arange(0, 1, 0.001)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_roc, pos_label=1)
    plt.plot(fpr,tpr, color='m', label = "ROC curve")
    plt.plot(guess_fp,guess_tp, color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    # plt.savefig('roc.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    result['roc'] = image_base64
    return result

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

        elif isinstance(model, (MLPClassifier, AdaBoostClassifier)):
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
            
        # SVM
        if isinstance(model, SVC):
            background = x_test[:20]
            x_explain = x_test[:5]

            if hasattr(model, "predict_proba"):
                f = lambda X: model.predict_proba(np.asarray(X, dtype=float))[:, 1]
                explainer = shap.KernelExplainer(f, background, link="logit")
                shap_values = explainer.shap_values(x_explain, nsamples=100)
            else:
                f = lambda X: model.predict(np.asarray(X, dtype=float)).astype(float)
                explainer = shap.KernelExplainer(f, background, link="identity")
                shap_values = explainer.shap_values(x_explain, nsamples=100)

            shap_values_for_plot = shap_values
            shap_data = x_explain

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

def predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names):
    model = load_model(model_path)
    x_test, y_test, y_pred, data_with_preds = predict_labels(model, model_path, data, label_column, pred_column)
    save_predictions(data_with_preds, data_path, output_name, task_dir)
    results = evaluate_model(y_test, y_pred, model, x_test)
    shap_result = explain_with_shap(model, x_test, feature_names)
    results.update(shap_result)
    lime_result = explain_with_lime(model, x_test, y_test, feature_names)
    results.update(lime_result)
    return results

def predict_stacking(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names):
    results = {
        "status": "success",
    }
    temp_dir = tempfile.mkdtemp()
    try:
        # 解壓 ZIP 到 temp 目錄
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        config_path = os.path.join(temp_dir, "stacking_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        # 載入 base models
        base_models = []
        for base_name in config.get("base_models", []):
            if base_name == "xgb":
                filename = f"base_{base_name}.json"
            elif base_name == "tabnet":
                filename = f"base_{base_name}.zip"
            else:
                filename = f"base_{base_name}.pkl"
            base_model_path = os.path.join(temp_dir, filename)
            model = load_model(base_model_path)
            base_models.append(model)
        # 載入 meta model
        meta_name = config.get("meta_model")
        if meta_name == "xgb":
            meta_file = f"meta_{meta_name}.json"
        elif meta_name == "tabnet":
            meta_file = f"meta_{meta_name}.zip"
        else:
            meta_file = f"meta_{meta_name}.pkl"
        meta_model_path = os.path.join(temp_dir, meta_file)
        meta_model = load_model(meta_model_path)
        # 載入資料
        x_test_df = data.drop(columns=[label_column], errors='ignore').copy()
        y_test = data[label_column].copy()
        # 轉換成 meta features：每個 base model 的 predict_proba[:,1]
        base_preds = []
        for model in base_models:
            if isinstance(model, TabNetClassifier):
                proba = model.predict_proba(x_test_df.to_numpy(dtype=np.float32))[:, 1]
            else:
                proba = model.predict_proba(x_test_df.values)[:, 1]
            base_preds.append(proba)
        X_meta = np.column_stack(base_preds)
        meta_feature_df = pd.DataFrame(X_meta, columns=[config.get("base_models", [])])
        save_predictions(meta_feature_df, data_path, "meta_feature", task_dir)
        # 最終預測
        y_pred = meta_model.predict(X_meta)
        data[pred_column] = y_pred
        # 儲存預測結果
        save_predictions(data, data_path, output_name, task_dir)
        # 評估
        results = evaluate_model(y_test, y_pred, meta_model, X_meta)
        shap_result = explain_with_shap(meta_model, X_meta, config.get("base_models", []))
        results.update(shap_result)
        lime_result = explain_with_lime(meta_model, X_meta, y_test, config.get("base_models", []))
        results.update(lime_result)
        return results
    except Exception as e:
        return {
            "status": "error",
            "message": f"{e}"
        }
    finally:
        shutil.rmtree(temp_dir)  # 清除暫存資料夾

def main(model_path, data_path, output_name, label_column, pred_column, task_dir):
    try:
        data = load_data(data_path)
        feature_names = data.drop(columns=[label_column], errors='ignore').columns.tolist()

        if model_path.endswith("zip"):
            try:
                with zipfile.ZipFile(args.model_path, 'r') as zip_ref:
                    if "stacking_config.json" in zip_ref.namelist():
                        # 是 stacking zip
                        results = predict_stacking(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names)
                    else:
                        # 是 tabnet 模型
                        results = predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names)
            except zipfile.BadZipFile:
                print(json.dumps({
                    "status": "error",
                    "message": f"Invalid zip file: {args.model_path}",
                }))
        else:
            results = predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, feature_names)
            
        results["task_dir"] = task_dir
        result_json_path = os.path.join(task_dir, "metrics.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        extract_base64_images_and_clean_json(task_dir, "metrics.json")
        print(json.dumps(results, indent=4, cls=NumpyEncoder))
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the trained model file (e.g., .json for XGBoost or .pkl for joblib models)")
    parser.add_argument('data_path', type=str, help="Path to input data file (.csv or .xlsx)")
    parser.add_argument('output_name', type=str, help="Output file name (without extension)")
    parser.add_argument('label_column', type=str, help="Name of the column that contains true labels")
    parser.add_argument("pred_column", type=str, help="Name of the column to store predictions")
    parser.add_argument("task_dir", type=str)
    
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_name, args.label_column, args.pred_column, args.task_dir)
