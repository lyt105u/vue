import json
import argparse
import os
from lightgbm import LGBMClassifier
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json

import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import numpy as np
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

def train_lgbm(x_train, y_train, x_val, y_val, model_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir):
    evals_result = {}

    y_train = np.asarray(y_train).astype(int)
    num_class = int(len(np.unique(y_train)))

    if num_class <= 2:
        objective = "binary"
        eval_metric = ["binary_logloss", "binary_error"]
        extra_params = {}
    else:
        objective = "multiclass"
        eval_metric = ["multi_logloss", "multi_error"]
        extra_params = {"num_class": num_class}

    lightgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        objective=objective,
        verbose=-1,
        **extra_params
    )

    lightgbm.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_names=["training", "validation"],
        eval_metric=eval_metric,
        callbacks=[lgb.log_evaluation(period=0), lgb.record_evaluation(evals_result)]
    )

    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(lightgbm, f"{task_dir}/{model_name}.pkl")

    return lightgbm, evals_result

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== (A) 決定二元視角 (one-vs-rest) 的 y_true_bin / y_score =====
    if num_class == 2:
        # 二元：正類視角仍是 class 1（同你原本）
        y_true_bin = y_test
        y_score = y_pred_proba[:, 1]
    else:
        focus_class = int(focus_class)
        y_true_bin = (y_test == focus_class).astype(int)
        y_score = y_pred_proba[:, focus_class]

    # ===== (B) 0.5 門檻的基本混淆矩陣/指標（前端舊格式）=====
    y_pred_bin = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    result = {
        "status": "success",
        "num_class": int(num_class),
        "focus_class": int(focus_class) if num_class > 2 else 1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_true_bin, y_pred_bin) * 100),
            "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0) * 100),
            "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0) * 100),
            "f1_score": float(f1_score(y_true_bin, y_pred_bin, zero_division=0) * 100),
        }
    }

    # ===== (C) threshold 掃描：維持你「舊版」行為 =====
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        # 舊版：用 class0 機率 + 反向規則
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
        # 多元：focus_class vs rest（正向規則）
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

    # ===== (D) ROC：二元/多元都用 y_true_bin vs y_score 畫一張 =====
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score, pos_label=1)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result['roc'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return result

def explain_with_shap(model, x_test, feature_names, focus_class=1):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)

        explainer = shap.Explainer(model, x_test)
        shap_values = explainer(x_test, check_additivity=False)

        # shap_values.values 可能是：
        # - binary: (n, p)
        # - multiclass: (n, p, k)
        values = shap_values.values
        shap_data = shap_values.data

        if hasattr(values, "ndim") and values.ndim == 3:
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= values.shape[2]:
                # 超界就用第一筆的預測類別
                focus_class = int(np.argmax(model.predict_proba(x_test[:1])[0]))
            shap_values_for_plot = values[:, :, focus_class]
            result["shap_focus_class"] = int(focus_class)
            result["shap_num_class"] = int(values.shape[2])
        else:
            shap_values_for_plot = values
            result["shap_focus_class"] = 1
            # 嘗試推 num_class（不強求）
            try:
                result["shap_num_class"] = int(model.predict_proba(x_test[:1]).shape[1])
            except Exception:
                result["shap_num_class"] = 2

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

def explain_with_lime(model, x_test, y_test, feature_names, focus_class=1, sample_index=0, num_features=10):
    result = {}
    try:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).astype(int)

        proba0 = model.predict_proba(x_test[:1])[0]
        num_class = int(len(proba0))
        class_names = [f"class_{i}" for i in range(num_class)]

        if num_class == 2:
            focus_class = 1
        else:
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= num_class:
                focus_class = int(np.argmax(model.predict_proba(x_test[[sample_index]])[0]))

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
            labels=[focus_class]
        )

        fig = lime_result.as_pyplot_figure(label=focus_class)
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        result["lime_example_0"] = lime_result.as_list(label=focus_class)
        result["lime_focus_class"] = int(focus_class)
        result["lime_num_class"] = int(num_class)

    except Exception as e:
        result["lime_error"] = str(e)

    return result

def plot_loss(evals_result):
    # 自動判斷 key
    if "binary_logloss" in evals_result["training"]:
        loss_key = "binary_logloss"
        ylabel = "Log Loss"
        title = "Training vs Validation Log Loss"
    elif "multi_logloss" in evals_result["training"]:
        loss_key = "multi_logloss"
        ylabel = "Multi-class Log Loss"
        title = "Training vs Validation Multi-class Log Loss"
    else:
        loss_key = list(evals_result["training"].keys())[0]
        ylabel = loss_key
        title = f"Training vs Validation {loss_key}"

    loss_train = evals_result["training"][loss_key]
    loss_val = evals_result["validation"][loss_key]
    epochs = range(1, len(loss_train) + 1)

    plt.figure()
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return image_base64


def plot_accuracy(evals_result):
    if "binary_error" in evals_result["training"]:
        err_key = "binary_error"
        title = "Training vs Validation Accuracy"
    elif "multi_error" in evals_result["training"]:
        err_key = "multi_error"
        title = "Training vs Validation Accuracy (Multi-class)"
    else:
        keys0 = list(evals_result["training"].keys())
        candidates = [k for k in keys0 if "error" in k.lower()]
        err_key = candidates[0] if candidates else keys0[0]
        title = f"Training vs Validation Accuracy ({err_key})"

    err_train = evals_result["training"][err_key]
    err_val = evals_result["validation"][err_key]

    acc_train = [1 - e for e in err_train]
    acc_val = [1 - e for e in err_val]
    epochs = range(1, len(acc_train) + 1)

    plt.figure()
    plt.plot(epochs, acc_train, label="Train Accuracy")
    plt.plot(epochs, acc_val, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return image_base64

def kfold_evaluation(x, y, cv_folds, model_name, n_estimators, learning_rate, max_depth, num_leaves,
                     feature_names, task_dir, focus_class=1):
    x = np.asarray(x)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=30)
    folds_result = []

    acc_list, rec_list, prec_list, f1_list = [], [], [], []
    total_tn = total_fp = total_fn = total_tp = 0

    for fold_index, (train_index, val_index) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model_fold_name = f"{model_name}_fold_{fold_index}"
        model, evals_result = train_lgbm(
            x_train, y_train, x_val, y_val,
            model_fold_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir
        )

        y_pred = model.predict(x_val)
        fold_eval = evaluate_model(y_val, y_pred, model, x_val, focus_class=focus_class)

        tn = fold_eval["confusion_matrix"]["true_negative"]
        fp = fold_eval["confusion_matrix"]["false_positive"]
        fn = fold_eval["confusion_matrix"]["false_negative"]
        tp = fold_eval["confusion_matrix"]["true_positive"]
        total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp

        acc_list.append(fold_eval["metrics"]["accuracy"])
        rec_list.append(fold_eval["metrics"]["recall"])
        prec_list.append(fold_eval["metrics"]["precision"])
        f1_list.append(fold_eval["metrics"]["f1_score"])

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
            "shap_error": shap_result.get("shap_error"),

            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0"),
            "lime_error": lime_result.get("lime_error"),

            "num_class": fold_eval.get("num_class"),
            "focus_class": fold_eval.get("focus_class"),

            "recall_80": fold_eval.get("recall_80"),
            "recall_85": fold_eval.get("recall_85"),
            "recall_90": fold_eval.get("recall_90"),
            "recall_95": fold_eval.get("recall_95"),
        })

    avg_result = {
        "accuracy": float(np.mean(acc_list)) if acc_list else 0.0,
        "recall": float(np.mean(rec_list)) if rec_list else 0.0,
        "precision": float(np.mean(prec_list)) if prec_list else 0.0,
        "f1_score": float(np.mean(f1_list)) if f1_list else 0.0,
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

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir):
    try:
        x, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

    if split_strategy == "train_test_split":
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=float(split_value), stratify=y, random_state=30
            )
            model, evals_result = train_lgbm(x_train, y_train, x_test, y_test, model_name, n_estimators, learning_rate, max_depth, num_leaves, task_dir)
            y_pred = model.predict(x_test)
            FOCUS = 1  # 你要關注的類別
            results = evaluate_model(y_test, y_pred, model, x_test, focus_class=FOCUS)
            results["loss_plot"] = plot_loss(evals_result)
            results["accuracy_plot"] = plot_accuracy(evals_result)

            shap_result = explain_with_shap(model, x_test, feature_names, focus_class=FOCUS)
            results.update(shap_result)

            lime_result = explain_with_lime(model, x_test, y_test, feature_names, focus_class=FOCUS)
            results.update(lime_result)
        except ValueError as e:
            print(json.dumps({
                "status": "error",
                "message": f"{e}",
            }))
            return

    elif split_strategy == "k_fold":
        try:
            results = kfold_evaluation(
                x, y,
                int(split_value),  # split_value 為 fold 數
                model_name,
                n_estimators, learning_rate, max_depth, num_leaves,
                feature_names,
                task_dir,
                focus_class=1
            )
        except ValueError as e:
            print(json.dumps({
                "status": "error",
                "message": f"{e}",
            }))
            return

    else:
        print(json.dumps({
            "status": "error",
            "message": "Unsupported split strategy"
        }))
        return

    results["task_dir"] = task_dir
    result_json_path = os.path.join(task_dir, "metrics.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
    extract_base64_images_and_clean_json(task_dir, "metrics.json")
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
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.n_estimators, args.learning_rate, args.max_depth, args.num_leaves, args.task_dir)
