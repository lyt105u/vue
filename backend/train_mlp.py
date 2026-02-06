import json
import argparse
import os
import joblib
from sklearn.neural_network import MLPClassifier
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
from sklearn.neural_network import MLPClassifier

def train_mlp(x_train, y_train, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter, n_iter_no_change, task_dir):
    hidden_layer_sizes = tuple(filter(lambda x: x is not None, [hidden_layer_1, hidden_layer_2, hidden_layer_3]))
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        solver='adam',
        alpha=0.0001,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=n_iter_no_change,
        verbose=False
    )
    mlp.fit(x_train, y_train)
    
    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(mlp, f"{task_dir}/{model_name}.pkl")

    return mlp, mlp.loss_curve_, mlp.validation_scores_

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== (1) 二元 vs 多元：決定二元視角 y_true_bin / y_score =====
    if num_class == 2:
        # 二元：沿用你舊版 threshold 的邏輯（用 class 0 機率掃）
        y_true_bin = y_test  # 0/1
        score_for_threshold = y_pred_proba[:, 0]  # 舊版用 x[0]
        # 用於 ROC：仍用正類(=1)機率
        roc_score = y_pred_proba[:, 1]
        focus_class_out = 1
    else:
        focus_class = int(focus_class)
        if focus_class < 0 or focus_class >= num_class:
            focus_class = int(np.argmax(model.predict_proba(x_test[:1])[0]))
        y_true_bin = (y_test == focus_class).astype(int)   # focus vs rest
        score_for_threshold = None                          # 多元用 y_score 正向掃
        roc_score = y_pred_proba[:, focus_class]
        focus_class_out = focus_class

    # ===== (2) 0.5 門檻的預設預測（供基本指標）=====
    if num_class == 2:
        # 舊版規則在 th=0.5 等價於：pred = 1 if p0 < 0.5
        y_pred_bin = np.array([0 if (p0 >= 0.5) else 1 for p0 in y_pred_proba[:, 0]], dtype=int)
    else:
        y_pred_bin = (roc_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    result = {
        "status": "success",
        "num_class": int(num_class),
        "focus_class": int(focus_class_out),
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

    # ===== (3) threshold 掃描：完全沿用舊版精神（先用 0~1 計算，最後輸出再 *100）=====
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []
    npv_list = []
    cm_list = []

    for th_int in range(1, 101):
        th = th_int * 0.01

        if num_class == 2:
            # 舊版：用 p(class0) >= th -> pred 0 else 1
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_threshold], dtype=int)
            y_true_for_cm = y_true_bin
        else:
            # 多元：focus 機率 >= th -> pred 1 else 0
            y_pred_th = (roc_score >= th).astype(int)
            y_true_for_cm = y_true_bin

        tn2, fp2, fn2, tp2 = confusion_matrix(y_true_for_cm, y_pred_th).ravel()
        specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
        npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

        accuracy_list.append(accuracy_score(y_true_for_cm, y_pred_th))
        precision_list.append(precision_score(y_true_for_cm, y_pred_th, zero_division=0))
        recall_list.append(recall_score(y_true_for_cm, y_pred_th, zero_division=0))
        f1_list.append(f1_score(y_true_for_cm, y_pred_th, zero_division=0))
        specificity_list.append(specificity)
        npv_list.append(npv)
        cm_list.append([tn2, fp2, fn2, tp2])

    for recall_standard in [80, 85, 90, 95]:
        target = recall_standard / 100.0
        candidates = [i for i, r in enumerate(recall_list) if r >= target]

        if candidates:
            best_i = max(candidates, key=lambda i: f1_list[i])
            tn2, fp2, fn2, tp2 = cm_list[best_i]
            precision_v = precision_list[best_i]
            recall_v = recall_list[best_i]
            f1_v = f1_list[best_i]
            acc_v = accuracy_list[best_i]
            spec_v = specificity_list[best_i]
            npv_v = npv_list[best_i]
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

    # ===== (4) ROC：二元/多元都用「正類/ focus 類」機率畫單張 =====
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, roc_score, pos_label=1)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result["roc"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return result

def explain_with_shap(model, x_test, feature_names, focus_class=1,
                      background_n=20, explain_n=5):
    result = {}
    try:
        x_test = np.asarray(x_test, dtype=np.float32)

        background = x_test[:background_n]
        explain_target = x_test[:explain_n]

        # 推類別數
        proba0 = model.predict_proba(explain_target[:1])[0]
        num_class = int(len(proba0))

        # 決定 focus class
        if num_class == 2:
            focus = 1
        else:
            focus = int(focus_class)
            if focus < 0 or focus >= num_class:
                focus = int(np.argmax(model.predict_proba(explain_target[:1])[0]))

        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(explain_target)

        # ---- 統一整理出 shap_values_for_plot: shape=(samples, features) ----
        if isinstance(shap_values, list):
            # list[len=num_class], each usually (n, p)
            shap_values_for_plot = np.asarray(shap_values[focus])
        else:
            # ndarray: 可能是 (n,p) 或 (n,p,k) 或 (k,n,p)
            arr = np.asarray(shap_values)

            if arr.ndim == 2:
                # (n, p)
                shap_values_for_plot = arr
            elif arr.ndim == 3:
                # 可能是 (k, n, p) 或 (n, p, k)
                if arr.shape[0] == num_class:
                    # (k, n, p)
                    shap_values_for_plot = arr[focus]
                elif arr.shape[-1] == num_class:
                    # (n, p, k)
                    shap_values_for_plot = arr[:, :, focus]
                else:
                    # 不明形狀，嘗試 squeeze 到 (n,p)
                    shap_values_for_plot = np.squeeze(arr)
            else:
                shap_values_for_plot = np.squeeze(arr)

        shap_values_for_plot = np.asarray(shap_values_for_plot)

        # 保底：確保是 (n, p)
        if shap_values_for_plot.ndim != 2:
            # 盡力把最後一維壓成 features
            shap_values_for_plot = shap_values_for_plot.reshape(shap_values_for_plot.shape[0], -1)

        shap_data = np.asarray(explain_target)

        # 平均重要度 -> (features,)
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)

        # 確保每個 val 都是 scalar
        result["shap_importance"] = {
            feature_names[i]: float(np.asarray(val).reshape(-1)[0])
            for i, val in enumerate(shap_importance)
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

        # meta（前端可不接）
        result["shap_focus_class"] = int(focus)
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
            focus = 1
        else:
            focus = int(focus_class)
            if focus < 0 or focus >= num_class:
                focus = int(np.argmax(model.predict_proba(x_test[[sample_index]])[0]))

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
            labels=[focus]
        )

        fig = lime_result.as_pyplot_figure(label=focus)
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        result["lime_example_0"] = lime_result.as_list(label=focus)
        result["lime_focus_class"] = int(focus)
        result["lime_num_class"] = int(num_class)

    except Exception as e:
        result["lime_error"] = str(e)

    return result

# plot train loss only
def plot_loss(evals_result):
    epochs = range(1, len(evals_result) + 1)

    plt.figure()
    plt.plot(epochs, evals_result, label='Train LogLoss')
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training Log Loss")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

# plot validation accuracy only
def plot_accuracy(evals_result):
    epochs = range(1, len(evals_result) + 1)

    plt.figure()
    plt.plot(epochs, evals_result, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def kfold_evaluation(X, y, cv_folds, model_name,
                     hidden_layer_1, hidden_layer_2, hidden_layer_3,
                     activation, learning_rate_init, max_iter, n_iter_no_change,
                     feature_names, task_dir, focus_class=1):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=30)
    folds_result = []

    acc_list, rec_list, prec_list, f1_list = [], [], [], []
    total_tn = total_fp = total_fn = total_tp = 0

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_fold_name = f"{model_name}_fold_{fold}"
        model, loss_curve, val_scores = train_mlp(
            X_train, y_train, model_fold_name,
            hidden_layer_1, hidden_layer_2, hidden_layer_3,
            activation, learning_rate_init, max_iter, n_iter_no_change, task_dir
        )

        y_pred = model.predict(X_test)
        fold_eval = evaluate_model(y_test, y_pred, model, X_test, focus_class=focus_class)

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

        loss_base64 = plot_loss(loss_curve)
        acc_base64 = plot_accuracy(val_scores)

        shap_result = explain_with_shap(model, X_test, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, X_test, y_test, feature_names, focus_class=focus_class)

        folds_result.append({
            "fold": int(fold),
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

def main(file_path, label_column, split_strategy, split_value, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter, n_iter_no_change, task_dir):
    # 處理空白的 layer
    def convert_arg(value):
        return None if value.lower() in ["null", "none", ""] else int(value)
    hidden_layer_1 = convert_arg(args.hidden_layer_1)
    hidden_layer_2 = convert_arg(args.hidden_layer_2)
    hidden_layer_3 = convert_arg(args.hidden_layer_3)

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
            model, loss_curve, val_scores = train_mlp(x_train, y_train, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter, n_iter_no_change, task_dir)
            y_pred = model.predict(x_test)
            results = evaluate_model(y_test, y_pred, model, x_test, focus_class=1)
            results["loss_plot"] = plot_loss(loss_curve)
            results["accuracy_plot"] = plot_accuracy(val_scores)
            shap_result = explain_with_shap(model, x_test, feature_names)
            results.update(shap_result)
            lime_result = explain_with_lime(model, x_test, y_test, feature_names)
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
                x, y, int(split_value),
                model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3,
                activation, learning_rate_init, max_iter, n_iter_no_change,
                feature_names, task_dir,
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
    parser.add_argument("hidden_layer_1", type=str)
    parser.add_argument("hidden_layer_2", type=str)
    parser.add_argument("hidden_layer_3", type=str)
    parser.add_argument("activation", type=str)
    parser.add_argument("learning_rate_init", type=float)
    parser.add_argument("max_iter", type=int)
    parser.add_argument("n_iter_no_change", type=int)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.hidden_layer_1, args.hidden_layer_2, args.hidden_layer_3, args.activation, args.learning_rate_init, args.max_iter, args.n_iter_no_change, args.task_dir)
