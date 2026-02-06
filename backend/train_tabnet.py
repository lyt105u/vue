import json
import argparse
import os
import numpy as np
import subprocess
import sys
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet==4.1.0"])
    from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json
from contextlib import contextmanager

import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
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

@contextmanager
def suppress_stdout():
    # 暫時關閉標準輸出來隱藏 `save_model()` 的訊息
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def train_tabnet(x_train, y_train, x_val, y_val, model_name, batch_size, max_epochs, patience, task_dir):
    tabnet = TabNetClassifier(verbose=0)    # verbose 隱藏輸出
    x_train_np = np.array(x_train, dtype=np.float32)    # 確保 x_train 和 y_train 是 numpy.ndarray，而非 object
    y_train_np = np.array(y_train, dtype=np.int64)
    x_val_np = np.array(x_val, dtype=np.float32)    # 確保 x_train 和 y_train 是 numpy.ndarray，而非 object
    y_val_np = np.array(y_val, dtype=np.int64)

    with suppress_stdout():
        tabnet.fit(
            x_train_np, y_train_np,
            eval_set=[(x_train_np, y_train_np), (x_val_np, y_val_np)],
            eval_name=["train", "valid"],
            eval_metric=["accuracy"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
    
    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        # 使用 suppress_stdout() 來隱藏 `save_model()` 的輸出
        with suppress_stdout():
            tabnet.save_model(f"{task_dir}/{model_name}")

    return tabnet, tabnet.history

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    """
    - 二元：維持原本輸出格式 + 原本 threshold 掃描邏輯（用 proba[:,0]）
    - 多元：轉成 focus_class vs rest 的二元視角（輸出格式不變，前端最省事）
    """
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test, dtype=np.float32)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== 0) 產生 binary 視角 y_true_bin / y_score(用於 ROC) =====
    if num_class == 2:
        y_true_bin = y_test
        # ROC 用 class 1 機率（習慣）
        y_score = y_pred_proba[:, 1]
    else:
        focus_class = int(focus_class)
        y_true_bin = (y_test == focus_class).astype(int)
        y_score = y_pred_proba[:, focus_class]

    # ===== 1) 用 0.5 做一個預設 tn/fp/fn/tp（給前端基本顯示）=====
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

    # ===== 2) threshold 掃描（用 0~1 計算，最後輸出再 *100；避免新舊差異）=====
    thresh_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    specificity_list = []
    npv_list = []
    cm_list = []

    if num_class == 2:
        # ---- 完全照你舊邏輯：用 proba[:,0]，p0>=th 判 0，否則 1 ----
        score0 = y_pred_proba[:, 0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score0], dtype=int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            thresh_list.append(th)
            accuracy_list.append(accuracy_score(y_test, y_pred_th))
            precision_list.append(precision_score(y_test, y_pred_th, zero_division=0))
            recall_list.append(recall_score(y_test, y_pred_th, zero_division=0))
            f1_list.append(f1_score(y_test, y_pred_th, zero_division=0))
            specificity_list.append(specificity)
            npv_list.append(npv)
            cm_list.append([tn2, fp2, fn2, tp2])
    else:
        # ---- 多元：focus_class vs rest ----
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score >= th).astype(int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            thresh_list.append(th)
            accuracy_list.append(accuracy_score(y_true_bin, y_pred_th))
            precision_list.append(precision_score(y_true_bin, y_pred_th, zero_division=0))
            recall_list.append(recall_score(y_true_bin, y_pred_th, zero_division=0))
            f1_list.append(f1_score(y_true_bin, y_pred_th, zero_division=0))
            specificity_list.append(specificity)
            npv_list.append(npv)
            cm_list.append([tn2, fp2, fn2, tp2])

    # ===== 3) 找 recall>=標準 的候選，取 f1 最大（完全對齊你舊版概念）=====
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

    # ===== 4) ROC（二元/多元都用 y_true_bin vs y_score 畫同一張）=====
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

def explain_with_shap(model, x_test, feature_names, focus_class=1, max_background=30, max_explain=10):
    result = {}
    try:
        x_test = np.asarray(x_test, dtype=np.float32)

        # 抽樣避免 KernelExplainer 太慢
        n = x_test.shape[0]
        bg_n = min(max_background, n)
        ex_n = min(max_explain, n)

        background = x_test[:bg_n]
        explain_data = x_test[:ex_n]

        # 類別數推斷
        num_class = int(model.predict_proba(background[:1]).shape[1])

        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(explain_data)

        # shap_values 可能是 list-of-arrays（每類一個）或 ndarray
        if isinstance(shap_values, list):
            # list: [ (ex_n, n_features), ... ] len=num_class
            if num_class == 2:
                class_idx = 1
            else:
                class_idx = int(focus_class)
                if class_idx < 0 or class_idx >= len(shap_values):
                    class_idx = int(np.argmax(model.predict_proba(explain_data[:1])[0]))
            shap_values_for_plot = np.asarray(shap_values[class_idx])
        else:
            shap_values = np.asarray(shap_values)
            # 可能是 (num_class, ex_n, n_features) 或 (ex_n, n_features, num_class)
            if shap_values.ndim == 3:
                if shap_values.shape[0] == num_class:
                    # (num_class, ex_n, n_features)
                    class_idx = 1 if num_class == 2 else int(focus_class)
                    shap_values_for_plot = shap_values[class_idx]
                else:
                    # (ex_n, n_features, num_class)
                    class_idx = 1 if num_class == 2 else int(focus_class)
                    shap_values_for_plot = shap_values[:, :, class_idx]
            else:
                shap_values_for_plot = shap_values

        if shap_values_for_plot.ndim != 2:
            raise ValueError(f"shap_values shape invalid: {shap_values_for_plot.shape}")

        # 平均重要度
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
        result["shap_importance"] = {feature_names[i]: float(v) for i, v in enumerate(shap_importance)}

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, explain_data, feature_names=feature_names, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

        result["shap_focus_class"] = int(focus_class) if num_class > 2 else 1
        result["shap_num_class"] = int(num_class)

    except Exception as e:
        result["shap_error"] = str(e)

    return result

def explain_with_lime(model, x_test, y_test, feature_names, focus_class=1, sample_index=0, num_features=10):
    result = {}
    try:
        x_test = np.asarray(x_test, dtype=np.float32)
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
        result["lime_sample_index"] = int(sample_index)

    except Exception as e:
        result["lime_error"] = str(e)

    return result

def plot_loss(evals_result):
    losses = evals_result['loss']
    epochs = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(epochs, losses, label='Train LogLoss')
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


def plot_accuracy(evals_result):
    acc_train = evals_result['train_accuracy']
    acc_val = evals_result['valid_accuracy']
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

def kfold_evaluation(X, y, cv_folds, model_name, batch_size, max_epochs, patience,
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
        model, evals_result = train_tabnet(
            X_train, y_train, X_test, y_test,
            model_fold_name, batch_size, max_epochs, patience, task_dir
        )

        X_test_np = np.asarray(X_test, dtype=np.float32)
        y_pred = model.predict(X_test_np)

        fold_eval = evaluate_model(y_test, y_pred, model, X_test_np, focus_class=focus_class)

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

        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)

        shap_result = explain_with_shap(model, X_test_np, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, X_test_np, y_test, feature_names, focus_class=focus_class)

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

def main(file_path, label_column, split_strategy, split_value, model_name, batch_size, max_epochs, patience, task_dir):
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
            model, evals_result = train_tabnet(x_train, y_train, x_test, y_test, model_name, batch_size, max_epochs, patience, task_dir)
            y_pred = model.predict(np.array(x_test, dtype=np.float32))  # 確保 x_test 在傳入前轉換為 numpy.float32
            results = evaluate_model(y_test, y_pred, model, np.array(x_test, dtype=np.float32), focus_class=1)
            results["loss_plot"] = plot_loss(evals_result)
            results["accuracy_plot"] = plot_accuracy(evals_result)
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
            results = kfold_evaluation(x, y, int(split_value), model_name, batch_size, max_epochs, patience, feature_names, task_dir, focus_class=1)
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
    parser.add_argument("batch_size", type=int)
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("patience", type=int)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.batch_size, args.max_epochs, args.patience, args.task_dir)
