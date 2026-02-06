from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss
)
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import io
import base64
import shap
from lime.lime_tabular import LimeTabularExplainer
import os
import joblib


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

def should_use_class_weight(y):
    counts = Counter(y)
    total = sum(counts.values())
    ratios = [v / total for v in counts.values()]
    return max(ratios) - min(ratios) > 0.25

def _build_lr(penalty="l2", alpha=1.0, max_iter=500, class_weight=None):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss='log_loss',
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            warm_start=True,
            class_weight=class_weight,
        )
    )


# =========================
# Step 2: train fold (OOF)
# =========================
def train_fold(X_train, y_train, X_val,
               penalty="l2", alpha=1.0, max_iter=500,
               focus_class=1):
    """
    回傳該 fold 的 validation 預測機率（作為 meta feature 的一欄）
    - binary: 回傳 proba[:,1]
    - multiclass: 回傳 proba[:,focus_class]
    """
    y_train = np.asarray(y_train).astype(int)
    num_class = _num_class_from_y(y_train)
    fc = _pick_focus_class(num_class, focus_class)

    class_weight = "balanced" if should_use_class_weight(y_train) else None
    logistic_reg = _build_lr(penalty, alpha, max_iter, class_weight=class_weight)
    logistic_reg.fit(X_train, y_train)

    proba = logistic_reg.predict_proba(X_val)
    return proba[:, fc]


# =========================
# Step 5/6: retrain models
# =========================
def retrain(X, y, feature_names, task_dir,
            split_value='1.0', save_model=True, model_role='base',
            penalty="l2", alpha=1.0, max_iter=500,
            focus_class=1):
    """
    split_value != '1.0'：train_test_split + 輸出評估/圖/SHAP/LIME
    split_value == '1.0'：全資料訓練，只存模型（results=None）
    """
    y = np.asarray(y).astype(int)
    num_class = _num_class_from_y(y)
    fc = _pick_focus_class(num_class, focus_class)

    class_weight = "balanced" if should_use_class_weight(y) else None
    logistic_reg = _build_lr(penalty, alpha, max_iter, class_weight=class_weight)

    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=float(split_value), stratify=y, random_state=30
        )

        # 你原本用 epoch loop 模擬 learning curve（雖然 SGDClassifier warm_start + fit 每次會重來一些狀態）
        loss_list_train, acc_list_train, loss_list_val, acc_list_val = [], [], [], []
        for _ in range(max_iter):
            logistic_reg.fit(X_train, y_train)

            y_train_proba = logistic_reg.predict_proba(X_train)
            y_val_proba = logistic_reg.predict_proba(X_test)

            loss_train = log_loss(y_train, y_train_proba)
            loss_val = log_loss(y_test, y_val_proba)

            acc_train = accuracy_score(y_train, logistic_reg.predict(X_train))
            acc_val = accuracy_score(y_test, logistic_reg.predict(X_test))

            loss_list_train.append(loss_train)
            acc_list_train.append(acc_train)
            loss_list_val.append(loss_val)
            acc_list_val.append(acc_val)

        evals_result = {
            "training": {"logloss": loss_list_train, "accuracy": acc_list_train},
            "validation": {"logloss": loss_list_val, "accuracy": acc_list_val},
        }

        y_pred = logistic_reg.predict(X_test)
        results = evaluate_model(y_test, y_pred, logistic_reg, X_test, focus_class=fc)

        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)

        shap_result = explain_with_shap(logistic_reg, X_test, feature_names, focus_class=fc)
        results.update(shap_result)

        lime_result = explain_with_lime(logistic_reg, X_test, y_test, feature_names, focus_class=fc)
        results.update(lime_result)
    else:
        logistic_reg.fit(X, y)
        results = None

    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(logistic_reg, f"{task_dir}/{model_role}_lr.pkl")

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    # - binary: 回傳 proba[:,1]
    # - multiclass: 回傳 proba[:,focus_class]
    model_path = f"{task_dir}/base_lr.pkl"
    model = joblib.load(model_path)

    proba = model.predict_proba(X)
    num_class = int(proba.shape[1])
    fc = _pick_focus_class(num_class, focus_class)
    return proba[:, fc]


# =========================
# Evaluation (one-vs-rest compatible)
# =========================
def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    # - binary：沿用你原本 key 結構
    # - multiclass：focus_class vs rest 的二元視角（前端最省事）
    # ※ threshold 掃描：binary 維持舊版用 proba[:,0] 反向判斷
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

    # 基礎 0.5
    y_pred_bin = (y_score_pos >= 0.5).astype(int)
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

    # ===== threshold 掃描（照你舊版行為）=====
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        score_for_threshold = proba[:, 0]  # 舊版：用 class0 機率反向判斷
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
# SHAP / LIME
# =========================
def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=2000):
    result = {}
    try:
        x_test = np.asarray(x_test, dtype=np.float32)

        n = x_test.shape[0]
        if n > max_samples:
            idx = np.random.RandomState(30).choice(n, size=max_samples, replace=False)
            x_bg = x_test[idx]
            x_plot = x_test[idx]
        else:
            x_bg = x_test
            x_plot = x_test

        # 先用 predict_proba 判斷類別數（Pipeline OK）
        proba0 = model.predict_proba(x_bg[:1])[0]
        num_class = int(len(proba0))
        fc = _pick_focus_class(num_class, focus_class) if num_class > 2 else 1

        # ✅ 用 callable 包起來：回傳「某一個 class 的機率」
        def f(X):
            p = model.predict_proba(X)
            return p[:, fc]

        # ✅ 這樣 Pipeline 也能被 SHAP 分析
        explainer = shap.Explainer(f, x_bg)
        shap_values = explainer(x_plot)

        values = shap_values.values
        data = getattr(shap_values, "data", None)
        if data is None:
            data = x_plot

        # values 通常會是 (n, features)
        values = np.asarray(values)
        if values.ndim != 2:
            # 保險：壓成 (n, features)
            values = values.reshape(values.shape[0], -1)

        # importance
        shap_importance = np.abs(values).mean(axis=0)
        shap_importance = np.asarray(shap_importance).reshape(-1)

        result["shap_focus_class"] = int(fc)
        result["shap_num_class"] = int(num_class)
        result["shap_importance"] = {
            str(feature_names[i]): float(shap_importance[i])
            for i in range(min(len(feature_names), len(shap_importance)))
        }

        # plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(values, data, feature_names=feature_names, show=False)
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
            model.predict_proba,
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
# Plots
# =========================
def plot_loss(evals_result):
    # 這份 LR evals_result 是你自建的：training/validation -> logloss
    loss_train = evals_result['training']['logloss']
    loss_val = evals_result['validation']['logloss']
    epochs = range(1, len(loss_train) + 1)

    plt.figure()
    plt.plot(epochs, loss_train, label='Train LogLoss')
    plt.plot(epochs, loss_val, label='Validation LogLoss')
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
    # 你這裡存的就是 accuracy，不是 error，所以不要做 1-e
    acc_train = evals_result['training']['accuracy']
    acc_val = evals_result['validation']['accuracy']
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