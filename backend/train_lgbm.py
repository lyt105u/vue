import json
import argparse
import os
from lightgbm import LGBMClassifier
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder

import pandas as pd
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

def train_lgbm(x_train, y_train, x_val, y_val, model_name, n_estimators, learning_rate, max_depth, num_leaves):
    evals_result = {}
    lightgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        verbose=-1
    )
    lightgbm.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_names=["training", "validation"],
        eval_metric=["binary_logloss", "binary_error"],
        callbacks=[lgb.log_evaluation, lgb.record_evaluation(evals_result)]
    )
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        joblib.dump(lightgbm, f"model/{model_name}.pkl")

    return lightgbm, evals_result

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

def explain_with_shap(model, x_test):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        
        # 使用 SHAP.Explainer，禁用加總檢查
        explainer = shap.Explainer(model, x_test)
        shap_values = explainer(x_test, check_additivity=False)

        # 處理 SHAP 輸出
        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_values_for_plot = shap_values.values[..., 1]  # binary classification class 1
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

    return result

def explain_with_lime(model, x_test, y_test):
    result = {}
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

def plot_loss(evals_result):
    logloss_train = evals_result['training']['binary_logloss']
    logloss_val = evals_result['validation']['binary_logloss']
    epochs = range(1, len(logloss_train) + 1)

    plt.figure()
    plt.plot(epochs, logloss_train, label='Train LogLoss')
    plt.plot(epochs, logloss_val, label='Validation LogLoss')
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training vs Validation Log Loss")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def plot_accuracy(evals_result):
    error_train = evals_result['training']['binary_error']
    error_val = evals_result['validation']['binary_error']
    acc_train = [1 - e for e in error_train]
    acc_val = [1 - e for e in error_val]
    epochs = range(1, len(acc_train) + 1)

    plt.figure()
    plt.plot(epochs, acc_train, label='Train Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def kfold_evaluation(x, y, cv_folds, model_name, n_estimators, learning_rate, max_depth, num_leaves):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    folds_result = []
    all_metrics = {
        "accuracy": [],
        "recall": [],
        "precision": [],
        "f1_score": [],
        "auc": []
    }
    total_tn = total_fp = total_fn = total_tp = 0
    y_test_all = []
    y_pred_proba_all = []

    for fold_index, (train_index, val_index) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model_fold_name = f"{model_name}_fold_{fold_index}"
        model, evals_result = train_lgbm(x_train, y_train, x_val, y_val, model_fold_name, n_estimators, learning_rate, max_depth, num_leaves)
        y_pred_proba = model.predict_proba(x_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc_score = auc(fpr, tpr)

        all_metrics["accuracy"].append(accuracy)
        all_metrics["recall"].append(recall)
        all_metrics["precision"].append(precision)
        all_metrics["f1_score"].append(f1)
        all_metrics["auc"].append(auc_score)

        plt.figure()
        plt.plot(fpr, tpr, color='m', label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], color='0', linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        roc_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close()

        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)
        shap_result = explain_with_shap(model, x_val)
        lime_result = explain_with_lime(model, x_val, y_val)

        folds_result.append({
            "fold": fold_index,
            "metrics": {
                "accuracy": accuracy * 100,
                "recall": recall * 100,
                "precision": precision * 100,
                "f1_score": f1 * 100,
                "auc": auc_score * 100,
            },
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
            "roc": roc_base64,
            "loss_plot": loss_base64,
            "accuracy_plot": acc_base64,
            "shap_plot": shap_result.get("shap_plot"),
            "shap_importance": shap_result.get("shap_importance"),
            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0")
        })

        y_test_all.extend(y_val)
        y_pred_proba_all.extend(y_pred_proba)

    avg_result = {
        "accuracy": float(np.mean(all_metrics["accuracy"])) * 100,
        "recall": float(np.mean(all_metrics["recall"])) * 100,
        "precision": float(np.mean(all_metrics["precision"])) * 100,
        "f1_score": float(np.mean(all_metrics["f1_score"])) * 100,
        "auc": float(np.mean(all_metrics["auc"])) * 100,
        "confusion_matrix": {
            "true_negative": total_tn,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "true_positive": total_tp,
        }
    }

    result = {
        "status": "success",
        "folds": folds_result,
        "average": avg_result
    }
    return result

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth, num_leaves):
    try:
        x, y = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

    if split_strategy == "train_test_split":
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=float(split_value), stratify=y, random_state=30
        )
        model, evals_result = train_lgbm(x_train, y_train, x_test, y_test, model_name, n_estimators, learning_rate, max_depth, num_leaves)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)
        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)
        shap_result = explain_with_shap(model, x_test)
        results.update(shap_result)
        lime_result = explain_with_lime(model, x_test, y_test)
        results.update(lime_result)

    elif split_strategy == "k_fold":
        results = kfold_evaluation(
            x, y,
            int(split_value),  # split_value 為 fold 數
            model_name,
            n_estimators, learning_rate, max_depth, num_leaves
        )

    else:
        print(json.dumps({
            "status": "error",
            "message": "Unsupported split strategy"
        }))
        return

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("label_column", type=str)
    parser.add_argument("split_strategy", type=str)
    parser.add_argument("split_value", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("num_leaves", type=int)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.n_estimators, args.learning_rate, args.max_depth, args.num_leaves)
