import os
import pandas as pd
import json
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import StratifiedKFold
import subprocess
import sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "shap==0.47.1"],
    stdout=subprocess.DEVNULL
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "lime==0.2.0.1"],
    stdout=subprocess.DEVNULL
)
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
# from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier

def prepare_data(file_path, label_column):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    if label_column not in df.columns:
        raise ValueError(f"Outcome column '{label_column}' not found in the dataset.")
    
    x = df.drop(columns=[label_column]).values
    y = df[label_column].values.astype(float)

    return x, y

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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

    # === SHAP Explain ===
    try:
        x_test = np.array(x_test, dtype=np.float32)
        # Pipeline (Logistic Regression)
        if isinstance(model, Pipeline) and "logisticregression" in model.named_steps:
            estimator = model.named_steps["logisticregression"]
            scaler = model.named_steps["standardscaler"]
            x_scaled = scaler.transform(x_test)

            explainer = shap.LinearExplainer(estimator, x_scaled, feature_perturbation="interventional")
            shap_values = explainer.shap_values(x_scaled)

            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[1]
            else:
                shap_values_for_plot = shap_values

            shap_data = x_scaled

        elif isinstance(model, MLPClassifier):
            # 只取少量資料以避免 KernelExplainer 過慢
            background = x_test[:10]
            explain_target = x_test[:5]

            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(explain_target)

            if isinstance(shap_values, list):
                shap_values_for_plot = np.array(shap_values[1])  # binary: class 1
            else:
                shap_values_for_plot = np.array(shap_values)

            shap_data = np.array(explain_target)

            # 修正 shape 避免 summary_plot 出錯
            if shap_values_for_plot.ndim == 3 and shap_values_for_plot.shape[-1] == 2:
                shap_values_for_plot = shap_values_for_plot[..., 1]

                # 其他模型（Tree-based or compatible）
        else:
            explainer = shap.Explainer(model, x_test)
            shap_values = explainer(x_test)

            if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                shap_values_for_plot = shap_values.values[..., 1]
            else:
                shap_values_for_plot = shap_values.values

            shap_data = shap_values.data

        # 平均重要度
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
        result["shap_importance"] = {
            f"feature_{i}": float(val) for i, val in enumerate(shap_importance)
        }

        # beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, shap_data, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

    except Exception as e:
        result["shap_error"] = str(e)

    # === LIME Explain (單一筆) ===
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=[f"feature_{i}" for i in range(x_test.shape[1])],
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

def kfold_evaluation(X, y, split_value, train_function):
    cv_folds = int(split_value)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)

    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    tpr_list = []
    fpr_list = []
    auc_list = []
    y_test_all = []
    y_pred_proba_all = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = train_function(X_train, y_train, '')

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        auc_list.append(auc_score)

        y_test_all.extend(y_test)
        y_pred_proba_all.extend(y_pred_proba)

    avg_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    avg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_auc = np.mean(auc_list)

    fpr_agg, tpr_agg, _ = roc_curve(y_test_all, y_pred_proba_all)
    plt.figure()
    plt.plot(fpr_agg, tpr_agg, color='m', label=f"ROC curve (AUC = {avg_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='0', linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    roc_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    result = {
        "status": "success",
        "metrics": {
            "accuracy": avg_accuracy * 100,
            "recall": avg_recall * 100,
            "precision": avg_precision * 100,
            "f1_score": avg_f1 * 100,
            "auc": avg_auc * 100
        },
        "confusion_matrix": {
            "true_negative": total_tn,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "true_positive": total_tp,
        },
        "roc": roc_base64
    }
    return result
