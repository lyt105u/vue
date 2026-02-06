# stacking_tabnet.py
# - binary / multiclass 自動切換（設計邏輯比照 stacking_xgb.py）
# - multiclass 用 focus_class vs rest 的二元視角輸出（維持前端最小改動）
# - OOF meta feature：binary -> proba[:,1]；multiclass -> proba[:,focus_class]
# - evaluate_model：binary 走你舊版 threshold 行為；multiclass 用 focus_class 機率正向掃描
# - SHAP：TabNet 用 KernelExplainer（較慢，做 subsample）
# - LIME：支援 multiclass，指定 labels=[focus_class]

import numpy as np
import os
import io
import sys
import base64
import shap
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

from contextlib import contextmanager
from pytorch_tabnet.tab_model import TabNetClassifier
from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn import metrics


# =========================
# Font
# =========================
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# Suppress stdout (TabNet save_model noisy)
# =========================
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =========================
# Utils
# =========================
def _num_class_from_y(y):
    y = np.asarray(y).astype(int)
    return int(len(np.unique(y)))

def _pick_focus_class(num_class, focus_class):
    if num_class <= 2:
        return 1
    fc = int(focus_class)
    if fc < 0 or fc >= num_class:
        fc = 0
    return fc

def _to_float32(x):
    return np.asarray(x, dtype=np.float32)

def _to_int64(y):
    return np.asarray(y, dtype=np.int64)


# =========================
# Step 2: train fold (OOF)
# =========================
def train_fold(X_train, y_train, X_val,
               batch_size=256, max_epochs=2, patience=10,
               focus_class=1):
    # """
    # 回傳：validation 預測機率 (作為 meta feature 的一欄)
    # - binary: proba[:,1]
    # - multiclass: proba[:,focus_class]
    # """
    y_train = np.asarray(y_train).astype(int)
    num_class = _num_class_from_y(y_train)
    fc = _pick_focus_class(num_class, focus_class)

    tabnet = TabNetClassifier(verbose=0)

    x_train_np = _to_float32(X_train)
    y_train_np = _to_int64(y_train)
    x_val_np = _to_float32(X_val)

    with suppress_stdout():
        tabnet.fit(
            x_train_np, y_train_np,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

    proba = tabnet.predict_proba(x_val_np)
    return proba[:, fc]


# =========================
# Step 5/6: retrain models
# =========================
def retrain(X, y, feature_names, task_dir,
            split_value='1.0', save_model=True, model_role='base',
            batch_size=256, max_epochs=2, patience=10,
            focus_class=1):
    # """
    # split_value != '1.0'：train_test_split + 評估 + 圖 + SHAP/LIME
    # split_value == '1.0'：全資料訓練，只存模型（results=None）
    # """
    y = np.asarray(y).astype(int)
    num_class = _num_class_from_y(y)
    fc = _pick_focus_class(num_class, focus_class)

    tabnet = TabNetClassifier(verbose=0)

    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=float(split_value), stratify=y, random_state=30
        )

        x_train_np = _to_float32(X_train)
        y_train_np = _to_int64(y_train)
        x_val_np = _to_float32(X_test)
        y_val_np = _to_int64(y_test)

        # TabNet history keys 依 eval_metric 而定；為了相容 binary/multiclass，
        # 這裡用 accuracy 當作 valid 指標（你原本就是 accuracy）
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

        evals_result = tabnet.history  # dict-like

        y_pred = tabnet.predict(x_val_np)
        results = evaluate_model(y_test, y_pred, tabnet, x_val_np, focus_class=fc)

        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)

        shap_result = explain_with_shap(tabnet, X_test, feature_names, focus_class=fc)
        results.update(shap_result)

        lime_result = explain_with_lime(tabnet, X_test, y_test, feature_names, focus_class=fc)
        results.update(lime_result)

    else:
        x_train_np = _to_float32(X)
        y_train_np = _to_int64(y)
        with suppress_stdout():
            tabnet.fit(
                x_train_np, y_train_np,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
        results = None

    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        with suppress_stdout():
            tabnet.save_model(f"{task_dir}/{model_role}_tabnet")

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    # """
    # - binary: 回傳 proba[:,1]
    # - multiclass: 回傳 proba[:,focus_class]
    # """
    model_path = f"{task_dir}/base_tabnet.zip"
    model = TabNetClassifier()
    model.load_model(model_path)

    X_np = _to_float32(X)
    proba = model.predict_proba(X_np)
    num_class = int(proba.shape[1])
    fc = _pick_focus_class(num_class, focus_class)
    return proba[:, fc]


# =========================
# Evaluation (one-vs-rest compatible)
# =========================
def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    # """
    # - binary：沿用舊版輸出 key 結構
    # - multiclass：focus_class vs rest 的二元視角（前端/上層最省事）
    # ※ threshold 掃描：
    #    - binary：照你舊版（用 proba[:,0] 反向判斷）
    #    - multiclass：用 y_score_pos = proba[:,focus_class] 正向掃描
    # """
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    proba = model.predict_proba(x_test)
    num_class = int(proba.shape[1])

    if num_class == 2:
        y_true_bin = y_test
        y_score_pos = proba[:, 1]
    else:
        fc = _pick_focus_class(num_class, focus_class)
        y_true_bin = (y_test == fc).astype(int)
        y_score_pos = proba[:, fc]

    # 基礎（0.5）
    y_pred_bin = (y_score_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    result = {
        "status": "success",
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
        },
        "num_class": int(num_class),
        "focus_class": int(focus_class) if num_class > 2 else 1
    }

    # ===== threshold 掃描（照你舊版行為）=====
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        score_for_threshold = proba[:, 0]  # 舊版用 class0 並反向判斷
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if (p0 >= th) else 1 for p0 in score_for_threshold], dtype=int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_th).ravel()
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
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score_pos >= th).astype(int)

            tn2, fp2, fn2, tp2 = confusion_matrix(y_true_bin, y_pred_th).ravel()
            specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
            npv = tn2 / (fn2 + tn2) if (fn2 + tn2) > 0 else 0

            accuracy_score_list.append(accuracy_score(y_true_bin, y_pred_th))
            precision_score_list.append(precision_score(y_true_bin, y_pred_th, zero_division=0))
            recall_score_list.append(recall_score(y_true_bin, y_pred_th, zero_division=0))
            f1_score_list.append(f1_score(y_true_bin, y_pred_th, zero_division=0))
            specificity_score_list.append(specificity)
            npv_score_list.append(npv)
            confusion_matrix_list.append([tn2, fp2, fn2, tp2])

    # 依舊輸出 recall_80/85/90/95，挑「recall >= 標準」中 f1 最大者
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
            "true_positive": int(tp2)
        }

    # ROC：二元或 focus-vs-rest 都可
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score_pos, pos_label=1)
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


# =========================
# SHAP / LIME (single-plot, focus_class)
# =========================
def explain_with_shap(model, x_test, feature_names, focus_class=1,
                      background_size=20, explain_size=5, max_evals=200):
    # """
    # TabNet 沒有 TreeExplainer：用 KernelExplainer（慢）
    # - binary：取 class1
    # - multiclass：取 focus_class
    # """
    result = {}
    try:
        x_test = _to_float32(x_test)
        n = x_test.shape[0]
        if n == 0:
            raise ValueError("x_test is empty")

        # 選背景/解釋樣本（固定前段，避免 random 造成重現性差）
        bg = x_test[:min(background_size, n)]
        ex = x_test[:min(explain_size, n)]

        proba0 = model.predict_proba(bg[:1])[0]
        num_class = int(len(proba0))
        fc = _pick_focus_class(num_class, focus_class)

        def f_predict(X):
            X = _to_float32(X)
            return model.predict_proba(X)

        explainer = shap.KernelExplainer(f_predict, bg)
        # shap_values: list (classes) or ndarray depending on shap version
        shap_values = explainer.shap_values(ex, nsamples=max_evals)

        # 統一成 (n_samples, n_features) for plotting
        if isinstance(shap_values, list):
            # list length = num_class
            shap_values_for_plot = shap_values[fc]
            result["shap_num_class"] = int(len(shap_values))
            result["shap_focus_class"] = int(fc)
        else:
            sv = np.asarray(shap_values)
            # 可能是 (n_samples, n_features, n_classes)
            if sv.ndim == 3:
                shap_values_for_plot = sv[:, :, fc]
                result["shap_num_class"] = int(sv.shape[2])
                result["shap_focus_class"] = int(fc)
            else:
                shap_values_for_plot = sv
                result["shap_num_class"] = int(num_class)
                result["shap_focus_class"] = int(fc)

        if shap_values_for_plot.ndim != 2:
            raise ValueError(f"shap_values_for_plot shape invalid: {shap_values_for_plot.shape}")

        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
        result["shap_importance"] = {feature_names[i]: float(v) for i, v in enumerate(shap_importance)}

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_for_plot, ex, feature_names=feature_names, show=False)
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


def explain_with_lime(model, x_test, y_test, feature_names,
                      focus_class=1, sample_index=0, num_features=10):
    result = {}
    try:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).astype(int)

        proba0 = model.predict_proba(_to_float32(x_test[:1]))[0]
        num_class = int(len(proba0))
        class_names = [f"class_{i}" for i in range(num_class)]
        fc = _pick_focus_class(num_class, focus_class)

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
            lambda X: model.predict_proba(_to_float32(X)),
            num_features=num_features,
            labels=[fc]
        )

        fig = lime_result.as_pyplot_figure(label=fc)
        fig.set_size_inches(10, 6)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
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
# Plots (TabNet history compatible)
# =========================
def plot_loss(evals_result):
    # """
    # TabNet history 通常沒有 logloss/mlogloss；常見 key: 'loss'
    # 若沒有就挑第一個數值序列當 fallback。
    # """
    try:
        if isinstance(evals_result, dict) and 'loss' in evals_result:
            losses = evals_result['loss']
            title = "Training Loss"
            ylabel = "Loss"
        else:
            # fallback: 找第一個 list/array 欄位
            if isinstance(evals_result, dict):
                k0 = next(k for k, v in evals_result.items() if isinstance(v, (list, tuple)) and len(v) > 0)
                losses = evals_result[k0]
                title = f"Training {k0}"
                ylabel = k0
            else:
                raise ValueError("evals_result format not supported")

        epochs = range(1, len(losses) + 1)

        plt.figure()
        plt.plot(epochs, losses, label='Train')
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

    except Exception:
        # 保底：不要讓整個流程因為 plot 爆掉
        return ""


def plot_accuracy(evals_result):
    """
    TabNet history 常見 key:
      - 'train_accuracy' / 'valid_accuracy'
    若沒有就 fallback 找 'accuracy' 相關 key。
    """
    try:
        if isinstance(evals_result, dict) and 'train_accuracy' in evals_result and 'valid_accuracy' in evals_result:
            acc_train = evals_result['train_accuracy']
            acc_val = evals_result['valid_accuracy']
            title = "Training vs Validation Accuracy"
        else:
            if not isinstance(evals_result, dict):
                raise ValueError("evals_result format not supported")

            # 找包含 accuracy 的 keys
            keys = list(evals_result.keys())
            train_keys = [k for k in keys if 'train' in k.lower() and 'acc' in k.lower()]
            valid_keys = [k for k in keys if ('valid' in k.lower() or 'val' in k.lower()) and 'acc' in k.lower()]
            if train_keys and valid_keys:
                acc_train = evals_result[train_keys[0]]
                acc_val = evals_result[valid_keys[0]]
                title = f"Training vs Validation Accuracy ({train_keys[0]}/{valid_keys[0]})"
            else:
                # 最後 fallback：用單一 accuracy 序列
                acc_key = next(k for k in keys if 'acc' in k.lower())
                acc_train = evals_result[acc_key]
                acc_val = None
                title = f"Accuracy ({acc_key})"

        epochs = range(1, len(acc_train) + 1)

        plt.figure()
        plt.plot(epochs, acc_train, label='Train Accuracy')
        if acc_val is not None:
            plt.plot(epochs, acc_val, label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
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

    except Exception:
        return ""