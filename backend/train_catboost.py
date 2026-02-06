# train_catboost.py

import json
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "catboost==1.2.8"],
    stdout=subprocess.DEVNULL
)
from catboost import CatBoostClassifier, Pool
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

# 字型設置
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def train_catboost(x_train, y_train, x_val, y_val,
                   model_name, iterations, learning_rate, depth, task_dir):
    y_train = np.asarray(y_train).astype(int)
    num_class = int(len(np.unique(y_train)))

    # CatBoost：二元 vs 多元
    if num_class <= 2:
        loss_function = "Logloss"
        eval_metric = "Logloss"
    else:
        loss_function = "MultiClass"
        eval_metric = "MultiClass"

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function=loss_function,
        eval_metric=eval_metric,
        verbose=0,
        use_best_model=True,
        custom_metric=["Accuracy"],  # 這個在 multiclass 也可用
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_val, y_val),
        plot=False
    )

    evals_result = model.get_evals_result()

    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(model, f"{task_dir}/{model_name}.pkl")

    return model, evals_result

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    # - binary：沿用舊邏輯（用 class0 機率掃 th：pred = 0 if p0>=th else 1）
    # - multiclass：focus_class vs rest（pred = 1 if p_focus>=th else 0）
    # - 輸出欄位維持舊格式：tn/fp/fn/tp + metrics + roc + recall_80..95
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== (A) 先做一個「0.5 的預設」confusion/metrics（維持舊輸出）=====
    if num_class == 2:
        # 二元：正類 = 1，分數用 proba[:,1]
        y_true_bin = y_test
        y_score = y_pred_proba[:, 1]
        y_pred_bin = (y_score >= 0.5).astype(int)
    else:
        # 多元：focus_class vs rest
        focus_class = int(focus_class)
        if focus_class < 0 or focus_class >= num_class:
            focus_class = int(np.argmax(y_pred_proba[0]))
        y_true_bin = (y_test == focus_class).astype(int)
        y_score = y_pred_proba[:, focus_class]
        y_pred_bin = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).ravel()

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

    # ===== (B) threshold 掃描（舊行為 + 多元支援）=====
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        # ✅ 完全照你舊版：用 p(class0) 掃 threshold
        score_for_th = y_pred_proba[:, 0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_th], dtype=int)

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
        # 多元：focus_class vs rest，用 y_true_bin / y_score 掃
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

    # ===== (C) 依 recall_80/85/90/95：在 recall>=門檻中選 f1 最大（完全舊策略）=====
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

    # ===== (D) ROC：二元/多元都畫同一張（用 y_true_bin vs y_score）=====
    if len(np.unique(y_true_bin)) < 2:
        result["roc_error"] = "ROC undefined: only one class present in y_true_bin."
    else:
        fpr, tpr, _ = roc_curve(y_true_bin, y_score, pos_label=1)
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

def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=2000):
    """
    CatBoost 多分類/二分類通用（最終修正版）：
    - 用 CatBoost 原生 SHAP：get_feature_importance(..., fstr_type="ShapValues")
    - 支援三種回傳：
        1) list (K 個 array，每個 (N, F+1))
        2) 2D array: (N, F+1) 或 (N, F)
        3) 3D array: (N, K, F+1)  <-- 你現在就是這種
    - 多分類取 focus_class 產生單張圖，前端不用改
    """
    result = {}
    try:
        x_test = np.asarray(x_test, dtype=np.float32)
        n = x_test.shape[0]
        if n == 0:
            raise ValueError("x_test is empty")

        if n > max_samples:
            idx = np.random.RandomState(30).choice(n, size=max_samples, replace=False)
            x_plot = x_test[idx]
        else:
            x_plot = x_test

        # 類別數
        proba0 = model.predict_proba(x_plot[:1])[0]
        num_class = int(len(proba0))
        result["shap_num_class"] = int(num_class)

        # Pool 帶 feature_names
        pool = Pool(x_plot, feature_names=feature_names)

        # ✅ 用 fstr_type
        shap_raw = model.get_feature_importance(pool, fstr_type="ShapValues")
        shap_raw_arr = np.asarray(shap_raw)

        # ---- 取 focus_class 的 shap_vals（目標 2D: (N, F 或 F+1)）----
        if isinstance(shap_raw, list):
            # 形式 1) list: len=K，每個 (N, F+1)
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= len(shap_raw):
                focus_class = int(np.argmax(proba0))
            shap_vals = np.asarray(shap_raw[focus_class])
            result["shap_focus_class"] = int(focus_class)

        else:
            # ndarray 形式 2/3
            if shap_raw_arr.ndim == 2:
                # 形式 2) (N, F+1) 或 (N, F)
                shap_vals = shap_raw_arr
                result["shap_focus_class"] = 1 if num_class == 2 else int(focus_class)

            elif shap_raw_arr.ndim == 3:
                # ✅ 形式 3) (N, K, F+1)  <-- 你這裡就是這個
                focus_class = int(focus_class)
                if focus_class < 0 or focus_class >= shap_raw_arr.shape[1]:
                    focus_class = int(np.argmax(proba0))
                shap_vals = shap_raw_arr[:, focus_class, :]   # -> (N, F+1)
                result["shap_focus_class"] = int(focus_class)

            else:
                raise ValueError(f"Unexpected shap_raw ndim={shap_raw_arr.ndim}, shape={shap_raw_arr.shape}")

        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim != 2:
            raise ValueError(f"Unexpected shap_vals ndim={shap_vals.ndim}, shape={shap_vals.shape}")

        # ---- 對齊 feature 維度 ----
        n_feat = int(x_plot.shape[1])  # 14

        # CatBoost 常見：(N, F+1) 最後一欄 base value
        if shap_vals.shape[1] == n_feat + 1:
            shap_values_for_plot = shap_vals[:, :n_feat]
        elif shap_vals.shape[1] == n_feat:
            shap_values_for_plot = shap_vals
        else:
            raise ValueError(
                f"SHAP feature dim mismatch: x has {n_feat} features, shap_vals has {shap_vals.shape[1]} cols"
            )

        # importance
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0).reshape(-1)
        result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

        # plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_for_plot,
            x_plot,
            feature_names=feature_names,
            show=False,
            max_display=min(30, n_feat)
        )
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
    learn = evals_result.get('learn', {})
    valid = evals_result.get('validation', {})

    # 自動找 loss key：優先 Logloss / MultiClass
    if 'Logloss' in learn and 'Logloss' in valid:
        loss_key = 'Logloss'
        ylabel = 'Log Loss'
        title = 'Training vs Validation Log Loss'
    elif 'MultiClass' in learn and 'MultiClass' in valid:
        loss_key = 'MultiClass'
        ylabel = 'Multi-class Loss'
        title = 'Training vs Validation Multi-class Loss'
    else:
        # 保險：挑 learn 裡第一個 key
        loss_key = list(learn.keys())[0]
        ylabel = loss_key
        title = f'Training vs Validation {loss_key}'

    loss_train = learn[loss_key]
    loss_val = valid[loss_key]
    epochs = range(1, len(loss_train) + 1)

    plt.figure()
    plt.plot(epochs, loss_train, label='Train')
    plt.plot(epochs, loss_val, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
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
    learn = evals_result.get('learn', {})
    valid = evals_result.get('validation', {})

    if 'Accuracy' in learn and 'Accuracy' in valid:
        acc_train = learn['Accuracy']
        acc_val = valid['Accuracy']
    else:
        # 保險：如果沒有 Accuracy，就不畫或回傳空（看你前端要不要顯示）
        return None

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

def kfold_evaluation(x, y, cv_folds, model_name, iterations, learning_rate, depth,
                     feature_names, task_dir, focus_class=1):
    x = np.asarray(x)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=30)
    folds_result = []

    acc_list, rec_list, prec_list, f1_list = [], [], [], []
    total_tn = total_fp = total_fn = total_tp = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, evals_result = train_catboost(
            x_train, y_train, x_val, y_val,
            f"{model_name}_fold_{fold}",
            iterations, learning_rate, depth, task_dir
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

        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)

        shap_result = explain_with_shap(model, x_val, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, x_val, y_val, feature_names, focus_class=focus_class)

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

def main(file_path, label_column, split_strategy, split_value, model_name, iterations, learning_rate, depth, task_dir):
    try:
        x, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return

    FOCUS_CLASS = 1
    if split_strategy == "train_test_split":
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=float(split_value), stratify=y, random_state=30)
        model, evals_result = train_catboost(x_train, y_train, x_test, y_test, model_name, iterations, learning_rate, depth, task_dir)
        y_pred = model.predict(x_test)
        result = evaluate_model(y_test, y_pred, model, x_test, focus_class=FOCUS_CLASS)
        result["loss_plot"] = plot_loss(evals_result)
        result["accuracy_plot"] = plot_accuracy(evals_result)
        result.update(explain_with_shap(model, x_test, feature_names, focus_class=FOCUS_CLASS))
        result.update(explain_with_lime(model, x_test, y_test, feature_names, focus_class=FOCUS_CLASS))
    elif split_strategy == "k_fold":
        result = kfold_evaluation(x, y, int(split_value), model_name, iterations, learning_rate, depth, feature_names, task_dir, focus_class=FOCUS_CLASS)
    else:
        print(json.dumps({"status": "error", "message": "Unsupported split strategy"}))
        return

    result["task_dir"] = task_dir
    with open(os.path.join(task_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)
    extract_base64_images_and_clean_json(task_dir, "metrics.json")
    print(json.dumps(result, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("label_column", type=str)
    parser.add_argument("split_strategy", type=str)
    parser.add_argument("split_value", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("iterations", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("depth", type=int)
    parser.add_argument("task_dir", type=str)
    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.iterations, args.learning_rate, args.depth, args.task_dir)
