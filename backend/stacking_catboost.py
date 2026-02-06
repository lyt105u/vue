from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn import metrics
import matplotlib.pyplot as plt
import base64
import io
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
import os
import joblib

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

def _build_cat(iterations=500, learning_rate=0.009, depth=6, num_class=2, use_best_model=False):
    """
    CatBoost:
    - 二元：eval_metric 用 Logloss，custom_metric 用 Accuracy
    - 多元：eval_metric 用 MultiClass，custom_metric 用 Accuracy
    """
    if num_class <= 2:
        eval_metric = "Logloss"
    else:
        eval_metric = "MultiClass"

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        eval_metric=eval_metric,
        custom_metric=["Accuracy"],
        verbose=0,
        use_best_model=use_best_model,
    )
    return model


# =========================
# Step 2: train fold (OOF)
# =========================
def train_fold(X_train, y_train, X_val, iterations=500, learning_rate=0.009, depth=6, focus_class=1):
    """
    回傳：validation 預測機率（作為 meta feature 的一欄）
    - binary: proba[:,1]
    - multiclass: proba[:,focus_class]
    """
    y_train = np.asarray(y_train).astype(int)
    num_class = _num_class_from_y(y_train)
    fc = _pick_focus_class(num_class, focus_class)

    model = _build_cat(iterations, learning_rate, depth, num_class=num_class, use_best_model=False)
    model.fit(X_train, y_train, plot=False)

    proba = model.predict_proba(X_val)
    return proba[:, fc]


# =========================
# Step 5/6: retrain models
# =========================
def retrain(X, y, feature_names, task_dir,
            split_value='1.0', save_model=True, model_role='base',
            iterations=500, learning_rate=0.009, depth=6,
            focus_class=1):
    """
    split_value != '1.0'：train_test_split + eval + plots + SHAP/LIME
    split_value == '1.0'：全資料訓練，只存模型（results=None）
    """
    y = np.asarray(y).astype(int)
    num_class = _num_class_from_y(y)
    fc = _pick_focus_class(num_class, focus_class)

    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=float(split_value), stratify=y, random_state=30
        )

        model = _build_cat(iterations, learning_rate, depth, num_class=num_class, use_best_model=True)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            plot=False
        )

        evals_result = model.get_evals_result()
        y_pred = model.predict(X_test)
        # CatBoost predict 可能回傳 shape (n,1)，轉成 1D int
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        results = evaluate_model(y_test, y_pred, model, X_test, focus_class=fc)

        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)

        shap_result = explain_with_shap(model, X_test, feature_names, focus_class=fc)
        results.update(shap_result)

        lime_result = explain_with_lime(model, X_test, y_test, feature_names, focus_class=fc)
        results.update(lime_result)
    else:
        model = _build_cat(iterations, learning_rate, depth, num_class=num_class, use_best_model=False)
        model.fit(X, y, plot=False)
        results = None

    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(model, f"{task_dir}/{model_role}_catboost.pkl")

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    """
    - binary: proba[:,1]
    - multiclass: proba[:,focus_class]
    """
    model_path = f"{task_dir}/base_catboost.pkl"
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
    - binary：沿用舊版 key 結構
    - multiclass：focus_class vs rest 的二元視角（前端/上層最省事）
    ※ threshold 掃描：採用你「舊版」邏輯（binary 用 proba[:,0] 反向判斷）
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
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    if num_class == 2:
        # 舊版：用 proba[:,0] 並反向判斷
        score_for_threshold = proba[:, 0]
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
        # multiclass：focus vs rest，用正類機率 y_score_pos 正向掃
        for th_int in range(1, 101):
            th = th_int * 0.01
            y_pred_th = (y_score_pos >= th).astype(int)

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

    # ===== 用舊版精神挑最佳（recall>=門檻下 f1 最大）=====
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

    # ===== ROC：binary / focus-vs-rest 都可直接畫 =====
    fpr, tpr, _ = metrics.roc_curve(y_true_bin, y_score_pos, pos_label=1)
    plt.plot(fpr, tpr, color='m', label="ROC curve")
    plt.plot(np.arange(0, 1, 0.001), np.arange(0, 1, 0.001), color='0', linestyle="-.")
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
# SHAP / LIME (single-plot, focus_class)
# =========================
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
        result["lime_focus_class"] = int(fc)
        result["lime_num_class"] = int(num_class)
        result["lime_sample_index"] = int(sample_index)
    except Exception as e:
        result["lime_error"] = str(e)
    return result


# =========================
# Plots (binary/multiclass compatible)
# =========================
def plot_loss(evals_result):
    """
    CatBoost get_evals_result() 通常是：
    - binary: {'learn': {'Logloss': [...]}, 'validation': {'Logloss': [...]}}
    - multiclass: {'learn': {'MultiClass': [...]}, 'validation': {'MultiClass': [...]}}
    """
    learn = evals_result.get('learn', {})
    valid = evals_result.get('validation', {})

    if 'Logloss' in learn:
        key = 'Logloss'
        ylabel = 'Log Loss'
        title = 'Training vs Validation Log Loss'
    elif 'MultiClass' in learn:
        key = 'MultiClass'
        ylabel = 'Multi-class Loss'
        title = 'Training vs Validation Multi-class Loss'
    else:
        # fallback
        key = list(learn.keys())[0]
        ylabel = key
        title = f'Training vs Validation {key}'

    train_loss = learn[key]
    val_loss = valid[key]
    epochs = range(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
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
    """
    你這裡用 custom_metric=["Accuracy"]，所以通常會有 Accuracy
    """
    learn = evals_result.get('learn', {})
    valid = evals_result.get('validation', {})

    if 'Accuracy' in learn:
        key = 'Accuracy'
    else:
        # fallback
        key = list(learn.keys())[0]

    acc_train = learn[key]
    acc_val = valid[key]
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