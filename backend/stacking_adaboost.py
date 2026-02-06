from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve
)
import matplotlib.pyplot as plt
import base64
import io
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib
from matplotlib import font_manager
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn import metrics


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

def _build_adaboost(n_estimators=100, learning_rate=1.0, depth=3):
    # SAMME.R 支援多分類（會有 predict_proba K 欄）
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm="SAMME.R",
        random_state=42
    )

def _get_num_class_from_model(model, X_any):
    proba = model.predict_proba(X_any[:1])[0]
    return int(len(proba))


# =========================
# Step 2: train fold (OOF)
# =========================
def train_fold(X_train, y_train, X_val, n_estimators=100, learning_rate=1.0, depth=3, focus_class=1):
    """
    回傳：該 fold 的 validation 預測機率（作為 meta feature 的一欄）
    - binary: 回傳 proba[:,1]
    - multiclass: 回傳 proba[:,focus_class]
    """
    y_train = np.asarray(y_train).astype(int)
    num_class = _num_class_from_y(y_train)
    fc = _pick_focus_class(num_class, focus_class)

    model = _build_adaboost(n_estimators, learning_rate, depth)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)
    return proba[:, fc]


# =========================
# Step 5/6: retrain models
# =========================
def retrain(X, y, feature_names, task_dir, split_value='1.0',
            save_model=True, model_role='base',
            n_estimators=100, learning_rate=1.0, depth=3,
            focus_class=1):
    """
    split_value != '1.0'：train_test_split + 評估 + 圖 + SHAP/LIME
    split_value == '1.0'：全資料訓練，只存模型（results=None）
    """
    y = np.asarray(y).astype(int)
    num_class = _num_class_from_y(y)
    fc = _pick_focus_class(num_class, focus_class)

    model = _build_adaboost(n_estimators, learning_rate, depth)

    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=float(split_value), stratify=y, random_state=30
        )
        model.fit(X_train, y_train)

        # staged metrics（支援多分類 log_loss/accuracy）
        evals_result = {
            "train": {"Logloss": [], "Accuracy": []},
            "validation": {"Logloss": [], "Accuracy": []}
        }
        for y_proba_train, y_proba_val in zip(model.staged_predict_proba(X_train), model.staged_predict_proba(X_test)):
            evals_result["train"]["Logloss"].append(log_loss(y_train, y_proba_train, labels=np.unique(y)))
            evals_result["validation"]["Logloss"].append(log_loss(y_test, y_proba_val, labels=np.unique(y)))

        for y_pred_train, y_pred_val in zip(model.staged_predict(X_train), model.staged_predict(X_test)):
            evals_result["train"]["Accuracy"].append(accuracy_score(y_train, y_pred_train))
            evals_result["validation"]["Accuracy"].append(accuracy_score(y_test, y_pred_val))

        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred, model, X_test, focus_class=fc)

        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)

        shap_result = explain_with_shap(model, X_test, feature_names, focus_class=fc)
        results.update(shap_result)

        lime_result = explain_with_lime(model, X_test, y_test, feature_names, focus_class=fc)
        results.update(lime_result)
    else:
        model.fit(X, y)
        results = None

    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(model, f"{task_dir}/{model_role}_adaboost.pkl")

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    """
    - binary: proba[:,1]
    - multiclass: proba[:,focus_class]
    """
    model_path = f"{task_dir}/base_adaboost.pkl"
    model = joblib.load(model_path)

    proba = model.predict_proba(X)
    num_class = int(proba.shape[1])
    fc = _pick_focus_class(num_class, focus_class)
    return proba[:, fc]


# =========================
# Evaluation (one-vs-rest compatible)
# =========================
def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    """
    - binary：沿用你原本的 key 結構
    - multiclass：focus_class vs rest 的二元視角（輸出維持舊版，前端最省事）
    ※ threshold 掃描：照你舊版 binary 行為（用 proba[:,0] 反向判斷）
    """
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

    # 0.5 基礎
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

    # threshold 掃描（保留你舊版行為）
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        score_for_threshold = proba[:, 0]  # 舊版用 proba[:,0]
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = np.array([0 if p0 >= th else 1 for p0 in score_for_threshold], dtype=int)
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

    # ROC（binary 或 focus-vs-rest 都可畫）
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score_pos, pos_label=1)
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


# =========================
# SHAP / LIME (focus_class compatible)
# =========================
def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=20, nsamples=100):
    """
    AdaBoost 用 KernelExplainer（較慢），這裡只取少量樣本。
    多分類：選 focus_class 的 SHAP matrix（維持輸出格式一致）
    """
    result = {}
    try:
        x_test = np.asarray(x_test)
        x_sample = x_test[:max_samples]

        # 先探 num_class
        proba0 = model.predict_proba(x_sample[:1])[0]
        num_class = int(len(proba0))
        fc = _pick_focus_class(num_class, focus_class) if num_class > 2 else 1

        explainer = shap.KernelExplainer(model.predict_proba, x_sample)
        shap_values = explainer.shap_values(x_sample, nsamples=nsamples)

        # shap_values 可能是 list[class] 或 ndarray
        if isinstance(shap_values, list):
            # list 長度 = num_class
            shap_matrix = shap_values[fc] if num_class > 2 else shap_values[1]
        else:
            # ndarray: (samples, features, classes) 或 (samples, features)
            if shap_values.ndim == 3:
                shap_matrix = shap_values[:, :, fc]
            else:
                shap_matrix = shap_values

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

        result["shap_num_class"] = int(num_class)
        result["shap_focus_class"] = int(fc if num_class > 2 else 1)
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
        fc = _pick_focus_class(num_class, focus_class) if num_class > 2 else 1

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
        result["lime_num_class"] = int(num_class)
        result["lime_focus_class"] = int(fc if num_class > 2 else 1)
        result["lime_sample_index"] = int(sample_index)
    except Exception as e:
        result["lime_error"] = str(e)
    return result


# =========================
# Plots
# =========================
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