# evaluate.py / predict.py (可直接覆蓋)
# usage:
#   python evaluate.py model.json input.xlsx result label pred task_dir --focus_class 2
#   python evaluate.py model.zip  input.xlsx result label pred task_dir --focus_class 2
#   (model.zip 若內含 stacking_config.json => 視為 stacking；否則視為 TabNet zip)

import argparse
import base64
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import font_manager
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from tool_train import NumpyEncoder, extract_base64_images_and_clean_json
from catboost import CatBoostClassifier, Pool

# 字型（中英）
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams["font.family"] = font_prop.get_name()
except Exception:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# Utils
# =========================
def _pick_focus_class(num_class, focus_class):
    # binary 固定 1
    if num_class <= 2:
        return 1
    fc = int(focus_class)
    if fc < 0 or fc >= num_class:
        fc = 0
    return fc


def _predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise ValueError(f"Model {model.__class__.__name__} does not support predict_proba")


def _is_tabnet_model(model, model_path=None):
    if isinstance(model, TabNetClassifier):
        return True
    if model_path and model_path.lower().endswith(".zip"):
        # 這裡只代表「可能是 TabNet」；stacking zip 會在 main 裡先分流
        return True
    return model.__class__.__name__.lower().startswith("tabnet")


def _to_model_input(model, model_path, X_df: pd.DataFrame):
    # TabNet 需要 float32 numpy
    if _is_tabnet_model(model, model_path):
        return X_df.to_numpy(dtype=np.float32)
    return X_df.values

def _is_catboost_model(model):
    # 不要用 "hasattr(get_feature_importance)" 來判斷，太寬鬆/容易誤判
    mod = getattr(model.__class__, "__module__", "").lower()
    name = getattr(model.__class__, "__name__", "").lower()
    return ("catboost" in mod) or name.startswith("catboost")


def load_model(model_path):
    mp = model_path.lower()
    if mp.endswith(".json"):
        model = XGBClassifier()
        model.load_model(model_path)
        return model
    if mp.endswith(".pkl"):
        return joblib.load(model_path)
    if mp.endswith(".zip"):
        # 注意：stacking zip 不會走到這裡（main 會先判斷）
        model = TabNetClassifier()
        model.load_model(model_path)
        return model

    print(json.dumps({"status": "error", "message": f"Unsupported model format: {model_path}"}))
    sys.exit(1)


def load_data(data_path):
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    if data_path.endswith(".xlsx"):
        return pd.read_excel(data_path)

    print(json.dumps({"status": "error", "message": "Unsupported file format."}))
    sys.exit(1)


def save_predictions(data, data_path, output_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)
    if data_path.endswith(".csv"):
        out = os.path.join(task_dir, f"{output_name}.csv")
        data.to_csv(out, index=False, encoding="utf-8-sig")
        return out
    if data_path.endswith(".xlsx"):
        out = os.path.join(task_dir, f"{output_name}.xlsx")
        data.to_excel(out, index=False)
        return out
    return None


def predict_labels(model, model_path, data, label_column, pred_column):
    X_df = data.drop(columns=[label_column], errors="ignore").copy()
    if label_column in data.columns:
        y_test = data[label_column].copy()
    else:
        # 沒有標籤就不評估，只輸出預測
        y_test = None

    X_in = _to_model_input(model, model_path, X_df)
    y_pred = model.predict(X_in)

    data[pred_column] = y_pred
    return X_in, y_test, y_pred, data


# =========================
# Evaluation (binary/multiclass compatible; focus_class vs rest)
# =========================
def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    """
    - binary：沿用你舊版行為（threshold 掃描用 proba[:,0] + 反向判斷）
    - multiclass：focus_class vs rest（二元視角輸出，前端最省事）
    - recall_80/85/90/95：達標取 f1 最大；不達標則取「recall 最大」（同 recall 再取 f1 最大）
      => 不會再整段變 0，且你會看到 recall 逼近或等於 100
    """
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    # --- predict_proba ---
    if not hasattr(model, "predict_proba"):
        return {
            "status": "error",
            "message": "model has no predict_proba()",
        }

    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # --- focus_class 合法化 ---
    if num_class <= 2:
        focus_used = 1
    else:
        focus_used = int(focus_class)
        if focus_used < 0 or focus_used >= num_class:
            focus_used = 0  # 超界就退回 0（你也可以改成 argmax(proba)）

    # --- 建立二元視角 y_true_bin / y_score_pos ---
    if num_class == 2:
        # binary：class1 當正類
        y_true_bin = y_test
        y_score_pos = y_pred_proba[:, 1]
    else:
        # multiclass：focus vs rest
        y_true_bin = (y_test == focus_used).astype(int)
        y_score_pos = y_pred_proba[:, focus_used]

    # --- 0.5 門檻的基本指標 ---
    y_pred_bin = (y_score_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    result = {
        "status": "success",
        "num_class": int(num_class),
        "focus_class": int(focus_used) if num_class > 2 else 1,
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

    # --- threshold 掃描（保留舊版 binary 行為）---
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        # ✅ 舊版：用 proba[:,0]，並反向判斷 => [0 if p0 >= th else 1]
        score_for_threshold = y_pred_proba[:, 0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_threshold], dtype=int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th, labels=[0, 1]).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            accuracy_score_list.append(accuracy_score(y_test, y_pred_th))
            precision_score_list.append(precision_score(y_test, y_pred_th, zero_division=0))
            recall_score_list.append(recall_score(y_test, y_pred_th, zero_division=0))
            f1_score_list.append(f1_score(y_test, y_pred_th, zero_division=0))
            specificity_score_list.append(specificity)
            npv_score_list.append(npv)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])

    else:
        # multiclass：focus vs rest，用 y_score_pos 正向掃
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score_pos >= th).astype(int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_th, labels=[0, 1]).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            accuracy_score_list.append(accuracy_score(y_true_bin, y_pred_th))
            precision_score_list.append(precision_score(y_true_bin, y_pred_th, zero_division=0))
            recall_score_list.append(recall_score(y_true_bin, y_pred_th, zero_division=0))
            f1_score_list.append(f1_score(y_true_bin, y_pred_th, zero_division=0))
            specificity_score_list.append(specificity)
            npv_score_list.append(npv)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])

    # --- 挑 best threshold：達標取 f1 最大；不達標取 recall 最大（同 recall 再取 f1 最大）---
    for recall_standard in [80, 85, 90, 95]:
        target = recall_standard / 100.0  # 0~1

        candidates = [i for i, r in enumerate(recall_score_list) if r >= target]

        if candidates:
            # 達標：取 f1 最大
            best_i = max(candidates, key=lambda i: f1_score_list[i])
        else:
            # ❗不達標：改取 recall 最大（你要的：recall 逼近/等於 100，而不是全部 0）
            if not recall_score_list:
                best_i = None
            else:
                max_recall = max(recall_score_list)
                fallback = [i for i, r in enumerate(recall_score_list) if r == max_recall]
                best_i = max(fallback, key=lambda i: f1_score_list[i]) if fallback else None

        if best_i is None:
            tn2 = fp2 = fn2 = tp2 = 0
            precision_v = recall_v = f1_v = acc_v = spec_v = npv_v = f2_v = 0
        else:
            tn2, fp2, fn2, tp2 = confusion_matrix_list[best_i]
            precision_v = precision_score_list[best_i]
            recall_v = recall_score_list[best_i]
            f1_v = f1_score_list[best_i]
            acc_v = accuracy_score_list[best_i]
            spec_v = specificity_score_list[best_i]
            npv_v = npv_score_list[best_i]
            f2_v = (5 * precision_v * recall_v) / (4 * precision_v + recall_v) if (4 * precision_v + recall_v) > 0 else 0

        result[f"recall_{recall_standard}"] = {
            "recall": float(recall_v * 100),
            "specificity": float(spec_v * 100),
            "precision": float(precision_v * 100),
            "npv": float(npv_v * 100),
            "f1_score": float(f1_v * 100),
            "f2_score": float(f2_v * 100),
            "accuracy": float(acc_v * 100),
            "true_negative": int(tn2),
            "false_positive": int(fp2),
            "false_negative": int(fn2),
            "true_positive": int(tp2),
        }

    # --- ROC：二元或 focus-vs-rest 都可畫 ---
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score_pos, pos_label=1)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    result["roc"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()

    return result

# =========================
# SHAP (binary/multiclass focus_class)
# =========================
# def _align_feature_names(feature_names, n_feat):
#     feature_names = list(feature_names) if feature_names is not None else []
#     if len(feature_names) == n_feat:
#         return feature_names
#     if len(feature_names) > n_feat:
#         return feature_names[:n_feat]
#     # len(feature_names) < n_feat：補齊
#     return feature_names + [f"f{i}" for i in range(len(feature_names), n_feat)]


def _align_feature_names(feature_names, n_features):
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    feature_names = list(feature_names)
    if len(feature_names) == n_features:
        return feature_names
    if len(feature_names) > n_features:
        return feature_names[:n_features]
    return feature_names + [f"feature_{i}" for i in range(len(feature_names), n_features)]


def _is_catboost_model(model):
    mod = getattr(model.__class__, "__module__", "").lower()
    name = getattr(model.__class__, "__name__", "").lower()
    return ("catboost" in mod) or name.startswith("catboost")


def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=2000, nsamples=100):
    # evaluate.py 通用 SHAP（支援二元/多元）：
    # - LightGBM：TreeExplainer(booster) 避免 feature_names_in_ 寫入錯誤
    # - CatBoost：原生 SHAP (get_feature_importance + Pool + fstr_type="ShapValues")
    # - AdaBoost：KernelExplainer(model.predict_proba)
    # - 其他：先 shap.Explainer；失敗 fallback KernelExplainer
    result = {}
    try:
        import numpy as np
        import shap
        import matplotlib.pyplot as plt
        import io, base64

        x_test = np.asarray(x_test)
        if x_test.ndim != 2:
            raise ValueError(f"x_test must be 2D, got shape={x_test.shape}")

        n, n_feat = x_test.shape
        if n == 0:
            raise ValueError("x_test is empty")

        # ✅ feature_names 對齊（修掉 15 vs 14）
        feature_names = _align_feature_names(feature_names, n_feat)

        # subsample（避免太慢）
        if n > max_samples:
            idx = np.random.RandomState(30).choice(n, size=max_samples, replace=False)
            x_sample = x_test[idx]
        else:
            x_sample = x_test

        # =========================================================
        # 0) LightGBM：強制 TreeExplainer(booster) 走這條，避開 shap.Explainer()
        #    以解你遇到的：can't set attribute 'feature_names_in_'
        # =========================================================
        def _is_lgbm_model(m):
            name = m.__class__.__name__.lower()
            return name.startswith("lgbm") or ("lightgbm" in str(type(m)).lower())

        if _is_lgbm_model(model):
            x_plot = np.asarray(x_sample, dtype=np.float32)

            if not hasattr(model, "predict_proba"):
                raise ValueError("LightGBM model has no predict_proba")

            proba0 = model.predict_proba(x_plot[:1])[0]
            num_class = int(len(proba0))
            result["shap_num_class"] = int(num_class)

            if num_class == 2:
                fc = 1
            else:
                fc = int(focus_class)
                if fc < 0 or fc >= num_class:
                    fc = int(np.argmax(proba0))
            result["shap_focus_class"] = int(fc)

            # 取 booster（不同版本屬性可能不同）
            booster = None
            if hasattr(model, "booster_"):
                booster = model.booster_
            elif hasattr(model, "_Booster"):
                booster = model._Booster

            if booster is None:
                # booster 拿不到就退回 KernelExplainer（慢但穩）
                bg = x_plot[: min(50, x_plot.shape[0])]
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                shap_values = explainer.shap_values(x_plot[: min(50, x_plot.shape[0])], nsamples=nsamples)
                x_used = x_plot[: min(50, x_plot.shape[0])]
            else:
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(x_plot)
                x_used = x_plot

            # shap_values 可能是 list 或 ndarray
            if isinstance(shap_values, list):
                # binary 有時 list 長度=2；multiclass list 長度=K
                if len(shap_values) > 1:
                    shap_matrix = np.asarray(shap_values[fc])
                else:
                    shap_matrix = np.asarray(shap_values[0])
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # 可能是 (N, K, F) 或 (N, F, K)
                if shap_values.shape[1] == num_class:
                    shap_matrix = shap_values[:, fc, :]
                else:
                    shap_matrix = shap_values[:, :, fc]
            else:
                shap_matrix = np.asarray(shap_values)

            # 對齊維度（有些版本會多一欄 base value）
            if shap_matrix.ndim != 2:
                raise ValueError(f"Unexpected LightGBM shap_matrix ndim={shap_matrix.ndim}, shape={shap_matrix.shape}")

            if shap_matrix.shape[1] == n_feat + 1:
                shap_matrix = shap_matrix[:, :n_feat]
            elif shap_matrix.shape[1] != n_feat:
                raise ValueError(f"LightGBM SHAP dim mismatch: x has {n_feat}, shap has {shap_matrix.shape[1]}")

            shap_importance = np.abs(shap_matrix).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, x_used, feature_names=feature_names, show=False)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

        # =========================================================
        # 1) CatBoost：只允許真的 CatBoost 進來
        # =========================================================
        if _is_catboost_model(model):
            from catboost import Pool

            x_plot = x_sample.astype(np.float32, copy=False)
            proba0 = model.predict_proba(x_plot[:1])[0]
            num_class = int(len(proba0))
            result["shap_num_class"] = int(num_class)

            pool = Pool(x_plot, feature_names=feature_names)
            shap_raw = model.get_feature_importance(pool, fstr_type="ShapValues")
            shap_raw_arr = np.asarray(shap_raw)

            if isinstance(shap_raw, list):
                fc = 1 if num_class == 2 else int(focus_class)
                if fc < 0 or fc >= len(shap_raw):
                    fc = int(np.argmax(proba0))
                shap_vals = np.asarray(shap_raw[fc])
                result["shap_focus_class"] = int(fc)
            else:
                if shap_raw_arr.ndim == 2:
                    shap_vals = shap_raw_arr
                    result["shap_focus_class"] = 1 if num_class == 2 else int(focus_class)
                elif shap_raw_arr.ndim == 3:
                    fc = 1 if num_class == 2 else int(focus_class)
                    if fc < 0 or fc >= shap_raw_arr.shape[1]:
                        fc = int(np.argmax(proba0))
                    shap_vals = shap_raw_arr[:, fc, :]
                    result["shap_focus_class"] = int(fc)
                else:
                    raise ValueError(f"Unexpected CatBoost shap_raw ndim={shap_raw_arr.ndim}, shape={shap_raw_arr.shape}")

            shap_vals = np.asarray(shap_vals)
            if shap_vals.ndim != 2:
                raise ValueError(f"Unexpected CatBoost shap_vals ndim={shap_vals.ndim}, shape={shap_vals.shape}")

            if shap_vals.shape[1] == n_feat + 1:
                shap_values_for_plot = shap_vals[:, :n_feat]
            elif shap_vals.shape[1] == n_feat:
                shap_values_for_plot = shap_vals
            else:
                raise ValueError(f"CatBoost SHAP dim mismatch: x has {n_feat}, shap has {shap_vals.shape[1]}")

            shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_for_plot, x_plot, feature_names=feature_names, show=False)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

        # =========================================================
        # 2) AdaBoost：強制 KernelExplainer
        # =========================================================
        from sklearn.ensemble import AdaBoostClassifier
        if isinstance(model, AdaBoostClassifier):
            x_plot = np.asarray(x_sample, dtype=float)

            proba0 = model.predict_proba(x_plot[:1])[0]
            num_class = int(len(proba0))
            result["shap_num_class"] = int(num_class)

            if num_class == 2:
                focus_used = 1
            else:
                focus_used = int(focus_class)
                if focus_used < 0 or focus_used >= num_class:
                    focus_used = int(np.argmax(proba0))
            result["shap_focus_class"] = int(focus_used)

            explainer = shap.KernelExplainer(model.predict_proba, x_plot)
            shap_values = explainer.shap_values(x_plot, nsamples=nsamples)

            if isinstance(shap_values, list):
                shap_matrix = np.asarray(shap_values[focus_used])
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_matrix = shap_values[:, :, focus_used]
            else:
                shap_matrix = np.asarray(shap_values)

            if shap_matrix.shape != x_plot.shape:
                raise ValueError(f"SHAP values shape {shap_matrix.shape} != x_plot shape {x_plot.shape}")

            shap_importance = np.abs(shap_matrix).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(v) for i, v in enumerate(shap_importance)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, x_plot, feature_names=feature_names, show=False)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

        # =========================================================
        # 3) 其他模型：shap.Explainer -> fallback KernelExplainer
        # =========================================================
        x_plot = np.asarray(x_sample, dtype=float)

        if hasattr(model, "predict_proba"):
            proba0 = model.predict_proba(x_plot[:1])[0]
            num_class = int(len(proba0))
        else:
            num_class = 2
        result["shap_num_class"] = int(num_class)

        if num_class == 2:
            focus_used = 1
        else:
            focus_used = int(focus_class)
            if focus_used < 0 or focus_used >= num_class:
                focus_used = 0
        result["shap_focus_class"] = int(focus_used)

        try:
            explainer = shap.Explainer(model, x_plot)
            sv = explainer(x_plot)
            values = np.asarray(sv.values)
            data = getattr(sv, "data", x_plot)

            if values.ndim == 2:
                shap_matrix = values
            elif values.ndim == 3:
                if values.shape[1] == n_feat:
                    shap_matrix = values[:, :, focus_used]
                elif values.shape[2] == n_feat:
                    shap_matrix = values[:, focus_used, :]
                else:
                    raise ValueError(f"Unsupported SHAP values shape={values.shape} with n_feat={n_feat}")
            else:
                raise ValueError(f"Unsupported SHAP ndim={values.ndim}, shape={values.shape}")

            if shap_matrix.shape[1] != n_feat:
                if shap_matrix.shape[1] == n_feat + 1:
                    shap_matrix = shap_matrix[:, :n_feat]
                else:
                    raise ValueError(f"SHAP dim mismatch: x has {n_feat}, shap has {shap_matrix.shape[1]}")

            shap_importance = np.abs(shap_matrix).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, data, feature_names=feature_names, show=False)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

        except Exception:
            if not hasattr(model, "predict_proba"):
                raise

            explainer = shap.KernelExplainer(model.predict_proba, x_plot)
            shap_values = explainer.shap_values(x_plot, nsamples=nsamples)

            if isinstance(shap_values, list):
                shap_matrix = np.asarray(shap_values[focus_used])
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_matrix = shap_values[:, :, focus_used]
            else:
                shap_matrix = np.asarray(shap_values)

            if shap_matrix.shape != x_plot.shape:
                raise ValueError(f"SHAP values shape {shap_matrix.shape} != x_plot shape {x_plot.shape}")

            shap_importance = np.abs(shap_matrix).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(v) for i, v in enumerate(shap_importance)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, x_plot, feature_names=feature_names, show=False)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

    except Exception as e:
        result["shap_error"] = str(e)
        return result

    
# =========================
# LIME (binary/multiclass focus_class)
# =========================
def explain_with_lime(model, x_test, y_test, feature_names, focus_class=1, sample_index=0, num_features=10):
    result = {}
    try:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).astype(int)

        proba0 = _predict_proba_safe(model, x_test[:1])[0]
        num_class = int(len(proba0))
        fc = _pick_focus_class(num_class, focus_class)
        class_names = [f"class_{i}" for i in range(num_class)]

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
            labels=[fc],
        )

        fig = lime_result.as_pyplot_figure(label=fc)
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

        result["lime_example_0"] = lime_result.as_list(label=fc)
        result["lime_focus_class"] = int(fc)
        result["lime_num_class"] = int(num_class)
        result["lime_sample_index"] = int(sample_index)

    except Exception as e:
        result["lime_error"] = str(e)

    return result


# =========================
# Predict single model
# =========================
def predict_single_model(model_path, data, label_column, pred_column, data_path, output_name, task_dir, focus_class=1):
    model = load_model(model_path)

    X_in, y_test, y_pred, data_with_preds = predict_labels(model, model_path, data, label_column, pred_column)
    save_predictions(data_with_preds, data_path, output_name, task_dir)

    # 沒有 label 就只輸出預測，不評估
    if y_test is None:
        return {"status": "success", "message": "No label column found; predictions saved.", "task_dir": task_dir}

    feature_names = data.drop(columns=[label_column, pred_column], errors='ignore').columns.tolist()

    results = evaluate_model(y_test, y_pred, model, X_in, focus_class=focus_class)
    results.update(explain_with_shap(model, X_in, feature_names, focus_class=focus_class))
    results.update(explain_with_lime(model, X_in, y_test, feature_names, focus_class=focus_class))
    return results


# =========================
# Predict stacking zip
# =========================
def predict_stacking(model_zip_path, data, label_column, pred_column, data_path, output_name, task_dir, focus_class=1):
    results = {"status": "success"}
    temp_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(model_zip_path, "r") as zf:
            zf.extractall(temp_dir)

        config_path = os.path.join(temp_dir, "stacking_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # focus_class：config 若有就覆蓋 CLI
        focus_class = int(config.get("focus_class", focus_class))

        base_names = config.get("base_models", [])
        meta_name = config.get("meta_model")

        # load base models
        base_models = []
        for base_name in base_names:
            if base_name == "xgb":
                filename = f"base_{base_name}.json"
            elif base_name == "tabnet":
                filename = f"base_{base_name}.zip"
            else:
                filename = f"base_{base_name}.pkl"

            base_path = os.path.join(temp_dir, filename)
            base_models.append(load_model(base_path))

        # load meta model
        if meta_name == "xgb":
            meta_file = f"meta_{meta_name}.json"
        elif meta_name == "tabnet":
            meta_file = f"meta_{meta_name}.zip"
        else:
            meta_file = f"meta_{meta_name}.pkl"

        meta_model_path = os.path.join(temp_dir, meta_file)
        meta_model = load_model(meta_model_path)

        # data
        X_df = data.drop(columns=[label_column], errors="ignore").copy()
        if label_column in data.columns:
            y_test = data[label_column].copy()
        else:
            y_test = None

        # meta features：每個 base model 取 proba[:, focus_class]（binary=[:,1]）
        base_preds = []
        for m in base_models:
            if isinstance(m, TabNetClassifier) or m.__class__.__name__.lower().startswith("tabnet"):
                proba = _predict_proba_safe(m, X_df.to_numpy(dtype=np.float32))
            else:
                proba = _predict_proba_safe(m, X_df.values)

            num_class = int(proba.shape[1])
            fc = _pick_focus_class(num_class, focus_class)
            base_preds.append(proba[:, fc])

        X_meta = np.column_stack(base_preds)

        # 存 meta_feature
        meta_feature_df = pd.DataFrame(X_meta, columns=base_names)
        save_predictions(meta_feature_df, data_path, "meta_feature", task_dir)

        # final predict
        y_pred = meta_model.predict(X_meta)
        data[pred_column] = y_pred
        save_predictions(data, data_path, output_name, task_dir)

        if y_test is None:
            return {"status": "success", "message": "No label column found; predictions saved.", "task_dir": task_dir}

        # evaluate + xai（meta model 的 feature_names 用 base_names）
        results = evaluate_model(y_test, y_pred, meta_model, X_meta, focus_class=focus_class)
        results.update(explain_with_shap(meta_model, X_meta, base_names, focus_class=focus_class))
        results.update(explain_with_lime(meta_model, X_meta, y_test, base_names, focus_class=focus_class))
        return results

    except Exception as e:
        return {"status": "error", "message": f"{e}"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================
# Main
# =========================
def main(model_path, data_path, output_name, label_column, pred_column, task_dir, focus_class=1):
    try:
        data = load_data(data_path)

        # zip 需要先判斷是不是 stacking
        if model_path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(model_path, "r") as zf:
                    if "stacking_config.json" in zf.namelist():
                        results = predict_stacking(
                            model_path, data, label_column, pred_column, data_path, output_name, task_dir, focus_class=focus_class
                        )
                    else:
                        # TabNet zip
                        results = predict_single_model(
                            model_path, data, label_column, pred_column, data_path, output_name, task_dir, focus_class=focus_class
                        )
            except zipfile.BadZipFile:
                results = {"status": "error", "message": f"Invalid zip file: {model_path}"}
        else:
            results = predict_single_model(
                model_path, data, label_column, pred_column, data_path, output_name, task_dir, focus_class=focus_class
            )

        results["task_dir"] = task_dir

        os.makedirs(task_dir, exist_ok=True)
        result_json_path = os.path.join(task_dir, "metrics.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)

        extract_base64_images_and_clean_json(task_dir, "metrics.json")
        print(json.dumps(results, indent=4, cls=NumpyEncoder, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"{e}"}))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to model file (.json/.pkl/.zip)")
    parser.add_argument("data_path", type=str, help="Path to input data (.csv/.xlsx)")
    parser.add_argument("output_name", type=str, help="Output file name (without extension)")
    parser.add_argument("label_column", type=str, help="True label column name")
    parser.add_argument("pred_column", type=str, help="Prediction output column name")
    parser.add_argument("task_dir", type=str, help="Output directory")
    parser.add_argument("--focus_class", type=int, default=1, help="For multiclass: focus_class vs rest (default=1)")

    args = parser.parse_args()
    main(
        args.model_path,
        args.data_path,
        args.output_name,
        args.label_column,
        args.pred_column,
        args.task_dir,
        focus_class=args.focus_class,
    )