from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import io
import base64
import shap
from lime.lime_tabular import LimeTabularExplainer
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

def _build_mlp(hidden_layer_1=128, hidden_layer_2=64, hidden_layer_3=None,
               activation="relu", learning_rate_init=0.001, max_iter=300, n_iter_no_change=50):
    hidden_layer_sizes = tuple(filter(lambda x: x is not None, [hidden_layer_1, hidden_layer_2, hidden_layer_3]))
    return MLPClassifier(
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


# =========================
# Step 2: train fold (OOF)
# =========================
def train_fold(X_train, y_train, X_val,
               hidden_layer_1=128, hidden_layer_2=64, hidden_layer_3=None,
               activation="relu", learning_rate_init=0.001, max_iter=300, n_iter_no_change=50,
               focus_class=1):
    """
    回傳：該 fold 的 validation 預測機率（作為 meta feature 的一欄）
    - binary: 回傳 proba[:,1]
    - multiclass: 回傳 proba[:,focus_class]
    """
    y_train = np.asarray(y_train).astype(int)
    num_class = _num_class_from_y(y_train)
    fc = _pick_focus_class(num_class, focus_class)

    mlp = _build_mlp(hidden_layer_1, hidden_layer_2, hidden_layer_3,
                     activation, learning_rate_init, max_iter, n_iter_no_change)
    mlp.fit(X_train, y_train)

    proba = mlp.predict_proba(X_val)
    return proba[:, fc]


# =========================
# Step 5/6: retrain models
# =========================
def retrain(X, y, feature_names, task_dir,
            split_value='1.0', save_model=True, model_role='base',
            hidden_layer_1=128, hidden_layer_2=64, hidden_layer_3=None,
            activation="relu", learning_rate_init=0.001, max_iter=300, n_iter_no_change=50,
            focus_class=1):
    """
    split_value != '1.0' 時：train_test_split，輸出評估 + 圖 + SHAP/LIME
    split_value == '1.0' 時：全資料訓練，只存模型（results=None）
    """
    y = np.asarray(y).astype(int)
    num_class = _num_class_from_y(y)
    fc = _pick_focus_class(num_class, focus_class)

    mlp = _build_mlp(hidden_layer_1, hidden_layer_2, hidden_layer_3,
                     activation, learning_rate_init, max_iter, n_iter_no_change)

    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=float(split_value), stratify=y, random_state=30
        )
        mlp.fit(X_train, y_train)

        # MLP 的 learning curve（注意：這是 train loss / validation score）
        loss_curve = getattr(mlp, "loss_curve_", None)
        val_scores = getattr(mlp, "validation_scores_", None)

        y_pred = mlp.predict(X_test)
        results = evaluate_model(y_test, y_pred, mlp, X_test, focus_class=fc)

        if loss_curve is not None:
            results["loss_plot"] = plot_loss(loss_curve)
        if val_scores is not None:
            results["accuracy_plot"] = plot_accuracy(val_scores)

        shap_result = explain_with_shap(mlp, X_test, feature_names, focus_class=fc)
        results.update(shap_result)

        lime_result = explain_with_lime(mlp, X_test, y_test, feature_names, focus_class=fc)
        results.update(lime_result)
    else:
        mlp.fit(X, y)
        results = None

    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        joblib.dump(mlp, f"{task_dir}/{model_role}_mlp.pkl")

    return results


def predict_full_meta(X, task_dir, focus_class=1):
    """
    - binary: 回傳 proba[:,1]
    - multiclass: 回傳 proba[:,focus_class]
    """
    model_path = f"{task_dir}/base_mlp.pkl"
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
        # 舊版：用 proba[:,0]，並反向判斷
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
        # multiclass：focus vs rest，用 y_score_pos 正向掃
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

    # 取 recall >= 門檻時的最佳 F1（舊版行為）
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

    # ROC：二元或 focus-vs-rest 都能畫（AUC 你如果需要再加）
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
            shap_values_for_plot = np.asarray(shap_values[focus])
        else:
            arr = np.asarray(shap_values)

            if arr.ndim == 2:
                shap_values_for_plot = arr
            elif arr.ndim == 3:
                if arr.shape[0] == num_class:
                    shap_values_for_plot = arr[focus]          # (k, n, p)
                elif arr.shape[-1] == num_class:
                    shap_values_for_plot = arr[:, :, focus]    # (n, p, k)
                else:
                    shap_values_for_plot = np.squeeze(arr)
            else:
                shap_values_for_plot = np.squeeze(arr)

        shap_values_for_plot = np.asarray(shap_values_for_plot)

        # 保底：確保是 (n, p)
        if shap_values_for_plot.ndim != 2:
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
def plot_loss(loss_curve):
    # train loss only
    epochs = range(1, len(loss_curve) + 1)

    plt.figure()
    plt.plot(epochs, loss_curve, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64


def plot_accuracy(val_scores):
    # validation accuracy only（MLP 的 validation_scores_ 是 score 值，不一定是 accuracy，但通常是）
    epochs = range(1, len(val_scores) + 1)

    plt.figure()
    plt.plot(epochs, val_scores, label='Validation Score')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Score")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64