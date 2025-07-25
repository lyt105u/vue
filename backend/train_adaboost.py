# train_adaboost.py

import json
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
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
import shap
from lime.lime_tabular import LimeTabularExplainer
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json
from sklearn.metrics import log_loss

# 字型設置
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def train_adaboost(x_train, y_train, x_val, y_val, model_name, n_estimators, learning_rate, depth, task_dir):
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm="SAMME.R",
        random_state=42
    )
    model.fit(x_train, y_train)
    evals_result = {
        "train": {"Logloss": [], "Accuracy": []},
        "validation": {"Logloss": [], "Accuracy": []}
    }
    for y_proba_train, y_proba_val in zip(model.staged_predict_proba(x_train), model.staged_predict_proba(x_val)):
        evals_result["train"]["Logloss"].append(log_loss(y_train, y_proba_train))
        evals_result["validation"]["Logloss"].append(log_loss(y_val, y_proba_val))
    for y_pred_train, y_pred_val in zip(model.staged_predict(x_train), model.staged_predict(x_val)):
        evals_result["train"]["Accuracy"].append(accuracy_score(y_train, y_pred_train))
        evals_result["validation"]["Accuracy"].append(accuracy_score(y_val, y_pred_val))
    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(model, f"{task_dir}/{model_name}.pkl")
    return model, evals_result

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

    y_pred_proba = model.predict_proba(x_test)
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    for th in range(1, 101):
        th *= 0.01
        y_pred = [0 if x[0] >= th else 1 for x in y_pred_proba]
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
        high_recall_precision = []
        high_recall_specificity = []
        high_recall_npv = []
        high_recall_confusion_matrix = []
        high_recall_f2score = []
        high_recall_recall = []

        for i, recall in enumerate(recall_score_list):
            if recall >= recall_standard:
                precision = precision_score_list[i]
                high_recall_f1score.append(f1_score_list[i])
                high_recall_accuracy.append(accuracy_score_list[i])
                high_recall_precision.append(precision)
                high_recall_recall.append(recall)
                high_recall_specificity.append(specificity_score_list[i])
                high_recall_npv.append(npv_score_list[i])
                high_recall_confusion_matrix.append(confusion_matrix_list[i])
                f2_score = (5 * precision * recall) / (4 * precision + recall)
                high_recall_f2score.append(f2_score)

        if high_recall_f1score:
            idx = np.argmax(high_recall_f1score)
            best_conf = high_recall_confusion_matrix[idx]
            result[f"recall_{recall_standard}"] = {
                "recall": high_recall_recall[idx],
                "specificity": high_recall_specificity[idx],
                "precision": high_recall_precision[idx],
                "npv": high_recall_npv[idx],
                "f1_score": high_recall_f1score[idx],
                "f2_score": high_recall_f2score[idx],
                "accuracy": high_recall_accuracy[idx],
                "true_negative": best_conf[0],
                "false_positive": best_conf[1],
                "false_negative": best_conf[2],
                "true_positive": best_conf[3]
            }

    # ROC plot
    y_pred_roc = [x[1] for x in y_pred_proba]
    fpr, tpr, _ = roc_curve(y_test, y_pred_roc)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot([0, 1], [0, 1], color='0', linestyle="-.")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result['roc'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return result

def explain_with_shap(model, x_test, feature_names):
    result = {}
    try:
        x_sample = np.array(x_test[:20])  # 20 samples

        explainer = shap.KernelExplainer(model.predict_proba, x_sample)
        shap_values = explainer.shap_values(x_sample, nsamples=100)

        # 如果結果是 3D array: (samples, features, classes)，則選取 class 1
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, 1]  # 選 class 1 的 SHAP 值
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]
        else:
            shap_matrix = shap_values  # 回歸 or 已是 2D array

        # 驗證形狀
        if shap_matrix.shape != x_sample.shape:
            raise ValueError(f"SHAP values shape {shap_matrix.shape} != x_sample shape {x_sample.shape}")

        # 特徵重要度計算
        shap_importance = np.abs(shap_matrix).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
        }

        # Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, x_sample, feature_names=feature_names, show=False)
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
            x_test[0], model.predict_proba, num_features=10
        )

        fig = lime_result.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        result["lime_example_0"] = lime_result.as_list()
    except Exception as e:
        result["lime_error"] = str(e)
    return result

def plot_loss(evals_result):
    logloss_train = evals_result['train']['Logloss']
    logloss_val = evals_result['validation']['Logloss']
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
    acc_train = evals_result['train']['Accuracy']
    acc_val = evals_result['validation']['Accuracy']
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

def kfold_evaluation(x, y, cv_folds, model_name, n_estimators, learning_rate, depth, feature_names, task_dir):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    folds_result = []
    total_tn = total_fp = total_fn = total_tp = 0
    all_metrics = {"accuracy": [], "recall": [], "precision": [], "f1_score": [], "auc": []}

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model, evals_result = train_adaboost(x_train, y_train, x_val, y_val, f"{model_name}_fold_{fold_index}", n_estimators, learning_rate, depth, task_dir)
        y_pred_proba = model.predict_proba(x_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc_score = auc(*roc_curve(y_val, y_pred_proba)[:2])

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
        shap_result = explain_with_shap(model, x_val, feature_names)
        lime_result = explain_with_lime(model, x_val, y_val, feature_names)

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
    return {"status": "success", "folds": folds_result, "average": avg_result}

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, depth, task_dir):
    try:
        x, y, feature_names = prepare_data(file_path, label_column)
        if split_strategy == "train_test_split":
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=float(split_value), stratify=y, random_state=30)
            model, evals_result = train_adaboost(x_train, y_train, x_test, y_test, model_name, n_estimators, learning_rate, depth, task_dir)
            y_pred = model.predict(x_test)
            result = evaluate_model(y_test, y_pred, model, x_test)
            result["loss_plot"] = plot_loss(evals_result)
            result["accuracy_plot"] = plot_accuracy(evals_result)
            result.update(explain_with_shap(model, x_test, feature_names))
            result.update(explain_with_lime(model, x_test, y_test, feature_names))
        elif split_strategy == "k_fold":
            result = kfold_evaluation(x, y, int(split_value), model_name, n_estimators, learning_rate, depth, feature_names, task_dir)
        else:
            print(json.dumps({"status": "error", "message": "Unsupported split strategy"}))
            return

        result["task_dir"] = task_dir
        with open(os.path.join(task_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
        extract_base64_images_and_clean_json(task_dir, "metrics.json")
        print(json.dumps(result, indent=4, cls=NumpyEncoder))
    except ValueError as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("label_column", type=str)
    parser.add_argument("split_strategy", type=str)
    parser.add_argument("split_value", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("depth", type=int)
    parser.add_argument("task_dir", type=str)
    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.n_estimators, args.learning_rate, args.depth, args.task_dir)
