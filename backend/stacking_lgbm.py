import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn import metrics
import matplotlib.pyplot as plt
import io
import base64
import os
import matplotlib
from matplotlib import font_manager

# 字型
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


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

def _build_lgbm(n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31, num_class=2):
    # LGBMClassifier: 二元/多元可直接靠 y 的類別數自動處理
    # （多分類時 predict_proba 會回 (n, num_class)）
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        verbose=-1
    )

def _eval_metrics_for_lgbm(num_class: int):
    # 你原本用 record_evaluation(evals_result)，所以 key 會跟 metric 名稱一致
    if num_class <= 2:
        return ["binary_logloss", "binary_error"]
    else:
        return ["multi_logloss", "multi_error"]


# =========================
# Step 2: train fold (OOF)
# =========================
def train_fold(X_train, y_train, X_val,
               n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31,
               focus_class=1):
    """
    回傳：該 fold 的 validation 預測機率（作為 meta feature 的一欄）
    - binary: 回傳 proba[:,1]
    - multiclass: 回傳 proba[:,focus_class]
    """
    y_train = np.asarray(y_train).astype(int)
    num_class = _num_class_from_y(y_train)
    fc = _pick_focus_class(num_class, focus_class)

    model = _build_lgbm(n_estimators, learning_rate, max_depth, num_leaves, num_class=num_class)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)
    return proba[:, fc]


# =========================
# Step 5/6: retrain models
# =========================
def retrain(X, y, feature_names, task_dir,
            split_value='1.0', save_model=True, model_role='base',
            n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31,
            focus_class=1):
    y = np.asarray(y).astype(int)
    num_class = _num_class_from_y(y)
    fc = _pick_focus_class(num_class, focus_class)

    model = _build_lgbm(n_estimators, learning_rate, max_depth, num_leaves, num_class=num_class)

    if str(split_value) != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=float(split_value), stratify=y, random_state=30
        )

        evals_result = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=["training", "validation"],
            eval_metric=_eval_metrics_for_lgbm(_num_class_from_y(y_train)),
            callbacks=[lgb.log_evaluation(period=0), lgb.record_evaluation(evals_result)]
        )

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
        joblib.dump(model, f"{task_dir}/{model_role}_lgbm.pkl")

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    model_path = f"{task_dir}/base_lgbm.pkl"
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
    - binary：輸出格式維持舊版
    - multiclass：focus_class vs rest（二元視角）
    - threshold 掃描：binary 維持你舊版（用 proba[:,0] 反向判斷）
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

    # ===== threshold 掃描（照你舊版行為）=====
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        score_for_threshold = proba[:, 0]  # 舊版用 class0 機率反向
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

    # recall_80/85/90/95：採用「舊版精神」：有候選就挑 f1 最大，沒有就 0
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

    # ROC：二元或 focus-vs-rest
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

        proba0 = model.predict_proba(x_bg[:1])[0]
        num_class = int(len(proba0))

        explainer = shap.Explainer(model, x_bg)
        shap_values = explainer(x_plot, check_additivity=False)

        values = shap_values.values
        data = shap_values.data

        if values.ndim == 3:
            fc = _pick_focus_class(values.shape[2], focus_class)
            values_for_plot = values[:, :, fc]
            result["shap_focus_class"] = int(fc)
            result["shap_num_class"] = int(values.shape[2])
        else:
            values_for_plot = values
            result["shap_focus_class"] = 1
            result["shap_num_class"] = int(num_class)

        shap_importance = np.abs(values_for_plot).mean(axis=0)
        result["shap_importance"] = {feature_names[i]: float(v) for i, v in enumerate(shap_importance)}

        plt.figure(figsize=(10, 6))
        shap.summary_plot(values_for_plot, data, feature_names=feature_names, show=False)
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
            fc = 1
        else:
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
    """
    evals_result 結構：
    - training/validation -> metric_name -> list
    metric_name 可能是 binary_logloss / multi_logloss
    """
    train_dict = evals_result.get("training", {})
    val_dict = evals_result.get("validation", {})

    if "binary_logloss" in train_dict:
        key = "binary_logloss"
        ylabel = "Log Loss"
        title = "Training vs Validation Log Loss"
    elif "multi_logloss" in train_dict:
        key = "multi_logloss"
        ylabel = "Multi-class Log Loss"
        title = "Training vs Validation Multi-class Log Loss"
    else:
        # 保險：挑第一個 key
        key = list(train_dict.keys())[0]
        ylabel = key
        title = f"Training vs Validation {key}"

    train_vals = train_dict[key]
    val_vals = val_dict.get(key, train_vals)
    epochs = range(1, len(train_vals) + 1)

    plt.figure()
    plt.plot(epochs, train_vals, label='Train')
    plt.plot(epochs, val_vals, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img


def plot_accuracy(evals_result):
    """
    metric_name 可能是 binary_error / multi_error
    """
    train_dict = evals_result.get("training", {})
    val_dict = evals_result.get("validation", {})

    if "binary_error" in train_dict:
        key = "binary_error"
        title = "Training vs Validation Accuracy"
    elif "multi_error" in train_dict:
        key = "multi_error"
        title = "Training vs Validation Accuracy (Multi-class)"
    else:
        key = list(train_dict.keys())[0]
        title = f"Training vs Validation Accuracy ({key})"

    train_err = train_dict[key]
    val_err = val_dict.get(key, train_err)

    acc_train = [1 - e for e in train_err]
    acc_val = [1 - e for e in val_err]
    epochs = range(1, len(acc_train) + 1)

    plt.figure()
    plt.plot(epochs, acc_train, label='Train Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img