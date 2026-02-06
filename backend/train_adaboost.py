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

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    """
    - 二元：沿用你舊版邏輯（用 proba[:,0] 來掃 threshold，並用反向規則預測 0/1）
    - 多元：focus_class vs rest 的 one-vs-rest（二元視角輸出，前端改動最小）
    """
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== 1) 先決定 binary 視角的 y_true_bin / y_score（用來算 ROC、掃 threshold）=====
    if num_class == 2:
        # 二元：正類固定 class 1
        y_true_bin = y_test
        # y_score 用 class 1 的機率（用於 ROC）
        y_score = y_pred_proba[:, 1]
        focus_used = 1
    else:
        # 多元：focus_class vs rest
        focus_used = int(focus_class)
        if focus_used < 0 or focus_used >= num_class:
            focus_used = int(np.argmax(model.predict_proba(x_test[:1])[0]))
        y_true_bin = (y_test == focus_used).astype(int)
        y_score = y_pred_proba[:, focus_used]

    # ===== 2) 用預設 0.5 做一個基本 confusion matrix / metrics（二元視角）=====
    y_pred_bin_05 = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin_05).ravel()

    result = {
        "status": "success",
        "num_class": int(num_class),
        "focus_class": int(focus_used),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_true_bin, y_pred_bin_05) * 100),
            "recall": float(recall_score(y_true_bin, y_pred_bin_05, zero_division=0) * 100),
            "precision": float(precision_score(y_true_bin, y_pred_bin_05, zero_division=0) * 100),
            "f1_score": float(f1_score(y_true_bin, y_pred_bin_05, zero_division=0) * 100),
        }
    }

    # ===== 3) threshold 掃描（recall_80/85/90/95）=====
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        # ★完全沿用你舊版：用 proba[:,0] 掃 threshold，並用反向規則預測
        score_for_threshold = y_pred_proba[:, 0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_threshold], dtype=int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            thresh_list.append(th)
            accuracy_score_list.append(accuracy_score(y_test, y_pred_th))
            precision_score_list.append(precision_score(y_test, y_pred_th, zero_division=0))
            recall_score_list.append(recall_score(y_test, y_pred_th, zero_division=0))
            f1_score_list.append(f1_score(y_test, y_pred_th, zero_division=0))
            specificity_score_list.append(specificity)
            npv_score_list.append(npv)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])
    else:
        # 多元：focus_class vs rest，用 y_true_bin / y_score
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score >= th).astype(int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            thresh_list.append(th)
            accuracy_score_list.append(accuracy_score(y_true_bin, y_pred_th))
            precision_score_list.append(precision_score(y_true_bin, y_pred_th, zero_division=0))
            recall_score_list.append(recall_score(y_true_bin, y_pred_th, zero_division=0))
            f1_score_list.append(f1_score(y_true_bin, y_pred_th, zero_division=0))
            specificity_score_list.append(specificity)
            npv_score_list.append(npv)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])

    # ===== 4) 找到每個 recall 標準下的「最佳 f1」threshold =====
    for recall_standard in [80, 85, 90, 95]:
        target = recall_standard / 100.0
        candidates = [i for i, r in enumerate(recall_score_list) if r >= target]

        if candidates:
            best_i = max(candidates, key=lambda i: f1_score_list[i])
            tn2, fp2, fn2, tp2 = confusion_matrix_list[best_i]
            precision_v = precision_score_list[best_i]
            recall_v = recall_score_list[best_i]
            f1_v = f1_score_list[best_i]
            acc_v = accuracy_score_list[best_i]
            spec_v = specificity_score_list[best_i]
            npv_v = npv_score_list[best_i]
            f2_v = (5 * precision_v * recall_v) / (4 * precision_v + recall_v) if (4 * precision_v + recall_v) > 0 else 0
        else:
            tn2 = fp2 = fn2 = tp2 = 0
            precision_v = recall_v = f1_v = acc_v = spec_v = npv_v = f2_v = 0

        result[f"recall_{recall_standard}"] = {
            "recall": recall_v * 100,
            "specificity": spec_v * 100,
            "precision": precision_v * 100,
            "npv": npv_v * 100,
            "f1_score": f1_v * 100,
            "f2_score": f2_v * 100,
            "accuracy": acc_v * 100,
            "true_negative": int(tn2),
            "false_positive": int(fp2),
            "false_negative": int(fn2),
            "true_positive": int(tp2)
        }

    # ===== 5) ROC（binary / multiclass 都用 y_true_bin vs y_score 畫一張）=====
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='m', label=f"ROC curve (AUC={roc_auc:.3f})")
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

    # 也把 auc 存進 metrics（前端不接也沒差）
    result["metrics"]["auc"] = float(roc_auc * 100)

    return result

def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=20, nsamples=100):
    result = {}
    try:
        x_test = np.asarray(x_test)
        x_sample = np.array(x_test[:max_samples])

        # 類別數
        num_class = int(model.predict_proba(x_sample[:1]).shape[1])

        # focus_class 合法化
        if num_class == 2:
            focus_used = 1
        else:
            focus_used = int(focus_class)
            if focus_used < 0 or focus_used >= num_class:
                focus_used = int(np.argmax(model.predict_proba(x_sample[:1])[0]))

        explainer = shap.KernelExplainer(model.predict_proba, x_sample)
        shap_values = explainer.shap_values(x_sample, nsamples=nsamples)

        # shap_values 可能是 list (每類一個) 或 ndarray
        if isinstance(shap_values, list):
            shap_matrix = shap_values[focus_used]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # (samples, features, classes)
            shap_matrix = shap_values[:, :, focus_used]
        else:
            shap_matrix = shap_values

        if shap_matrix.shape != x_sample.shape:
            raise ValueError(f"SHAP values shape {shap_matrix.shape} != x_sample shape {x_sample.shape}")

        shap_importance = np.abs(shap_matrix).mean(axis=0)
        result["shap_importance"] = {feature_names[i]: float(v) for i, v in enumerate(shap_importance)}

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, x_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

        result["shap_focus_class"] = int(focus_used)
        result["shap_num_class"] = int(num_class)

    except Exception as e:
        result["shap_error"] = str(e)

    return result

def explain_with_lime(model, x_test, y_test, feature_names, focus_class=1, sample_index=0, num_features=10):
    result = {}
    try:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).astype(int)

        proba0 = model.predict_proba(x_test[:1])[0]
        num_class = int(len(proba0))
        class_names = [f"class_{i}" for i in range(num_class)]

        if num_class == 2:
            focus_used = 1
        else:
            focus_used = int(focus_class)
            if focus_used < 0 or focus_used >= num_class:
                focus_used = int(np.argmax(model.predict_proba(x_test[[sample_index]])[0]))

        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test[sample_index],
            model.predict_proba,
            num_features=num_features,
            labels=[focus_used]
        )

        fig = lime_result.as_pyplot_figure(label=focus_used)
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        result["lime_example_0"] = lime_result.as_list(label=focus_used)
        result["lime_focus_class"] = int(focus_used)
        result["lime_num_class"] = int(num_class)

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

def kfold_evaluation(x, y, cv_folds, model_name, n_estimators, learning_rate, depth, feature_names, task_dir, focus_class=1):
    x = np.asarray(x)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=30)
    folds_result = []

    acc_list, rec_list, prec_list, f1_list, auc_list = [], [], [], [], []
    total_tn = total_fp = total_fn = total_tp = 0

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, evals_result = train_adaboost(
            x_train, y_train, x_val, y_val,
            f"{model_name}_fold_{fold_index}",
            n_estimators, learning_rate, depth, task_dir
        )

        y_pred = model.predict(x_val)
        fold_eval = evaluate_model(y_val, y_pred, model, x_val, focus_class=focus_class)

        tn = fold_eval["confusion_matrix"]["true_negative"]
        fp = fold_eval["confusion_matrix"]["false_positive"]
        fn = fold_eval["confusion_matrix"]["false_negative"]
        tp = fold_eval["confusion_matrix"]["true_positive"]

        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        acc_list.append(fold_eval["metrics"]["accuracy"])
        rec_list.append(fold_eval["metrics"]["recall"])
        prec_list.append(fold_eval["metrics"]["precision"])
        f1_list.append(fold_eval["metrics"]["f1_score"])
        auc_list.append(fold_eval["metrics"].get("auc", 0))

        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)
        shap_result = explain_with_shap(model, x_val, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, x_val, y_val, feature_names, focus_class=focus_class)

        folds_result.append({
            "fold": int(fold_index),
            "metrics": fold_eval.get("metrics"),
            "confusion_matrix": fold_eval.get("confusion_matrix"),
            "roc": fold_eval.get("roc"),
            "loss_plot": loss_base64,
            "accuracy_plot": acc_base64,
            "shap_plot": shap_result.get("shap_plot"),
            "shap_importance": shap_result.get("shap_importance"),
            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0"),
            "recall_80": fold_eval.get("recall_80"),
            "recall_85": fold_eval.get("recall_85"),
            "recall_90": fold_eval.get("recall_90"),
            "recall_95": fold_eval.get("recall_95"),
            "num_class": fold_eval.get("num_class"),
            "focus_class": fold_eval.get("focus_class"),
        })

    avg_result = {
        "accuracy": float(np.mean(acc_list)) if acc_list else 0.0,
        "recall": float(np.mean(rec_list)) if rec_list else 0.0,
        "precision": float(np.mean(prec_list)) if prec_list else 0.0,
        "f1_score": float(np.mean(f1_list)) if f1_list else 0.0,
        "auc": float(np.mean(auc_list)) if auc_list else 0.0,
        "confusion_matrix": {
            "true_negative": int(total_tn),
            "false_positive": int(total_fp),
            "false_negative": int(total_fn),
            "true_positive": int(total_tp),
        }
    }

    return {
        "status": "success",
        "num_class": int(folds_result[0].get("num_class", 2)) if folds_result else 2,
        "focus_class": int(focus_class),
        "folds": folds_result,
        "average": avg_result
    }

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, depth, task_dir):
    try:
        x, y, feature_names = prepare_data(file_path, label_column)
        if split_strategy == "train_test_split":
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=float(split_value), stratify=y, random_state=30)
            model, evals_result = train_adaboost(x_train, y_train, x_test, y_test, model_name, n_estimators, learning_rate, depth, task_dir)
            y_pred = model.predict(x_test)
            focus_class = 1  # 你要的正類（多分類 focus）
            result = evaluate_model(y_test, y_pred, model, x_test, focus_class=focus_class)
            result["loss_plot"] = plot_loss(evals_result)
            result["accuracy_plot"] = plot_accuracy(evals_result)
            result.update(explain_with_shap(model, x_test, feature_names, focus_class=focus_class))
            result.update(explain_with_lime(model, x_test, y_test, feature_names, focus_class=focus_class))
        elif split_strategy == "k_fold":
            result = kfold_evaluation(x, y, int(split_value), model_name, n_estimators, learning_rate, depth, feature_names, task_dir, focus_class=1)
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
