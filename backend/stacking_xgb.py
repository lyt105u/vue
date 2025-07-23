import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn import metrics
import matplotlib.pyplot as plt
import base64
import io
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

def train_fold(X_train, y_train, X_val, n_estimators=100, learning_rate=0.300000012, max_depth=6):
    # Train XGBClassifier on a given fold and return validation predictions and the model.

    # Args:
    #   X_train: np.ndarray of training features for this fold
    #   y_train: np.ndarray of training labels for this fold
    #   X_val: np.ndarray of validation features for this fold
    #   params: dict of XGBClassifier parameters

    # Returns:
    #   val_preds: np.ndarray of predicted probabilities for class 1 on X_val
    #   model: trained XGBClassifier instance
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                        gamma=0, device='cuda', importance_type=None,
                        interaction_constraints='', learning_rate=learning_rate,
                        max_delta_step=0, max_depth=max_depth, min_child_weight=1,
                        monotone_constraints='()', n_estimators=n_estimators, n_jobs=72,
                        num_parallel_tree=1, random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                        validate_parameters=1, verbosity=None, eval_metric=['logloss', 'error'],
                        tree_method='gpu_hist',
                        predictor='gpu_predictor')
    xgb.fit(X_train, y_train)
    val_preds = xgb.predict_proba(X_val)[:, 1]
    return val_preds

def retrain(X, y, feature_names, task_dir, split_value=1.0, save_model=True, model_role='base', n_estimators=100, learning_rate=0.300000012, max_depth=6):
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                        gamma=0, device='cuda', importance_type=None,
                        interaction_constraints='', learning_rate=learning_rate,
                        max_delta_step=0, max_depth=max_depth, min_child_weight=1,
                        monotone_constraints='()', n_estimators=n_estimators, n_jobs=72,
                        num_parallel_tree=1, random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                        validate_parameters=1, verbosity=None, eval_metric=['logloss', 'error'],
                        tree_method='gpu_hist',
                        predictor='gpu_predictor')
    if split_value != '1.0':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=float(split_value), stratify=y, random_state=30)
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        evals_result = xgb.evals_result()
        y_pred = xgb.predict(X_test)
        results = evaluate_model(y_test, y_pred, xgb, X_test)
        results["loss_plot"] = plot_loss(evals_result)
        results["accuracy_plot"] = plot_accuracy(evals_result)
        shap_result = explain_with_shap(xgb, X_test, feature_names)
        results.update(shap_result)
        lime_result = explain_with_lime(xgb, X_test, y_test, feature_names)
        results.update(lime_result)
    else:
        xgb.fit(X, y)
        results = None
    if save_model:
        os.makedirs(task_dir, exist_ok=True)
        xgb.save_model(f"{task_dir}/{model_role}_xgb.json")
    return results

def evaluate_model(y_test, y_pred, model, x_test):
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result = {
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred) * 100),
            "recall": float(recall_score(y_test, y_pred) * 100),
            "precision": float(precision_score(y_test, y_pred) * 100),
            "f1_score": float(f1_score(y_test, y_pred) * 100),
        }
    }

    # recall 分析
    y_pred_proba = model.predict_proba(x_test)
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []
    for th in range(1,101):
        th *= 0.01
        y_pred = [0 if (x[0] >= th) else 1 for x in y_pred_proba]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn+fp) if (tn + fp) > 0 else 0
        npv = tn / (fn+tn) if (fn + tn) > 0 else 0
        thresh_list.append(th)
        accuracy_score_list.append(accuracy_score(y_test, y_pred) * 100)
        precision_score_list.append(precision_score(y_test, y_pred, zero_division=0) * 100)
        recall_score_list.append(recall_score(y_test, y_pred, zero_division=0) * 100)
        f1_score_list.append(f1_score(y_test, y_pred, zero_division=0) * 100)
        specificity_score_list.append(specificity * 100)
        npv_score_list.append(npv * 100)
        confusion_matrix_list.append([tn, fp, fn, tp])

    recall_standard_list = [80, 85, 90, 95]
    for recall_standard in recall_standard_list:
        high_recall_f1score = []
        high_recall_accuracy = []
        high_recall_recall = []
        high_recall_precision = []
        high_recall_specificity = []
        high_recall_npv = []
        high_recall_confusion_matrix = []
        high_recall_f2score = []
        for index, recall in enumerate(recall_score_list):
            if recall >= recall_standard:
                precision = precision_score_list[index]
                high_recall_f1score.append(f1_score_list[index])
                high_recall_accuracy.append(accuracy_score_list[index])
                high_recall_precision.append(precision_score_list[index])
                high_recall_recall.append(recall_score_list[index])
                high_recall_specificity.append(specificity_score_list[index])
                high_recall_npv.append(npv_score_list[index])
                high_recall_confusion_matrix.append(confusion_matrix_list[index])
                f2_score = (5 * precision * recall) / (4 * precision + recall)
                high_recall_f2score.append(f2_score)
        
        if high_recall_f1score:
            # 檢查是否有符合的 Recall 值
            high_recall_best_f1_score_index = np.argmax(high_recall_f1score)
            best_recall = high_recall_recall[high_recall_best_f1_score_index]
            best_specificity = high_recall_specificity[high_recall_best_f1_score_index]
            best_precision = high_recall_precision[high_recall_best_f1_score_index]
            best_npv = high_recall_npv[high_recall_best_f1_score_index]
            best_f1 = high_recall_f1score[high_recall_best_f1_score_index]
            best_f2 = high_recall_f2score[high_recall_best_f1_score_index]
            best_accuracy = high_recall_accuracy[high_recall_best_f1_score_index]
            best_confusion_matrix = high_recall_confusion_matrix[high_recall_best_f1_score_index]
        else:
            # 如果沒有符合 Recall ≥ 80, 85, 90, 95，填入 0
            best_recall = 0
            best_specificity = 0
            best_precision = 0
            best_npv = 0
            best_f1 = 0
            best_f2 = 0
            best_accuracy = 0
            best_confusion_matrix = [0, 0, 0, 0]

        key_name = f"recall_{recall_standard}"
        result[key_name] = {
            "recall": best_recall,
            "specificity": best_specificity,
            "precision": best_precision,
            "npv": best_npv,
            "f1_score": best_f1,
            "f2_score": best_f2,
            "accuracy": best_accuracy,
            "true_negative": best_confusion_matrix[0],
            "false_positive": best_confusion_matrix[1],
            "false_negative": best_confusion_matrix[2],
            "true_positive": best_confusion_matrix[3]
        }

    # plot
    y_pred_roc = [x[1] for x in y_pred_proba]
    guess_tp = np.arange(0, 1, 0.001)
    guess_fp = np.arange(0, 1, 0.001)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_roc, pos_label=1)
    plt.plot(fpr,tpr, color='m', label = "ROC curve")
    plt.plot(guess_fp,guess_tp, color='0', linestyle="-.")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    # plt.savefig('roc.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    result['roc'] = image_base64
    return result

def explain_with_shap(model, x_test, feature_names):
    result = {}
    try:
        x_test = np.array(x_test, dtype=np.float32)
        explainer = shap.Explainer(model, x_test)
        shap_values = explainer(x_test)

        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_values_for_plot = shap_values.values[..., 1]
        else:
            shap_values_for_plot = shap_values.values

        shap_data = shap_values.data

        # 平均重要度
        shap_importance = np.abs(shap_values_for_plot).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
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

    except Exception as e:
        result["shap_error"] = str(e)

    return result

def explain_with_lime(model, x_test, y_test, feature_names):
    result = {}
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=["class_0", "class_1"],
            discretize_continuous=True,
        )

        lime_result = lime_explainer.explain_instance(
            x_test[0],  # 第0筆樣本
            model.predict_proba,
            num_features=10
            # num_features=len(x_test[0])
        )

        # 儲存圖片到記憶體 buffer
        fig = lime_result.as_pyplot_figure()
        fig.set_size_inches(10, 6)  # 改變圖的尺寸
        fig.tight_layout()  # 自動調整 layout
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)  # 關閉圖避免重疊

        # 同時保留文字格式
        result["lime_example_0"] = lime_result.as_list()

    except Exception as e:
        result["lime_error"] = str(e)
        
    return result

def plot_loss(evals_result):
    logloss_train = evals_result['validation_0']['logloss']
    logloss_val = evals_result['validation_1']['logloss']
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
    error_train = evals_result['validation_0']['error']
    error_val = evals_result['validation_1']['error']
    acc_train = [1 - e for e in error_train]
    acc_val = [1 - e for e in error_val]
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
