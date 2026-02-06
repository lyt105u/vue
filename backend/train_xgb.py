# usage: python train_xgb.py upload/高醫訓練csv.csv label train_test_split 0.8 xgb_model 100 0.300000012 6 <task_dir>
# usage: python train_xgb.py upload/高醫訓練csv.csv label k_fold 2 "" 100 0.300000012 6 <task_dir>
import json
import argparse
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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
import base64
import io

from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import subprocess
import sys
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

def train_xgb(x_train, y_train, x_val, y_val, model_name, n_estimators, learning_rate, max_depth, task_dir):
    evals_result = {}

    num_class = int(len(np.unique(y_train)))
    if num_class <= 2:
        objective = "binary:logistic"
        eval_metric = ["logloss", "error"]
        xgb_kwargs = {}
    else:
        objective = "multi:softprob"
        eval_metric = ["mlogloss", "merror"]
        xgb_kwargs = {"num_class": num_class}

    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                        gamma=0, device='cuda', importance_type=None,
                        interaction_constraints='', learning_rate=learning_rate,
                        max_delta_step=0, max_depth=max_depth, min_child_weight=1,
                        monotone_constraints='()', n_estimators=n_estimators, n_jobs=72,
                        num_parallel_tree=1, random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                        validate_parameters=1, verbosity=None,
                        tree_method='gpu_hist',
                        predictor='gpu_predictor',
                        objective=objective,
                        eval_metric=eval_metric,
                        **xgb_kwargs)
    xgb.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False
    )

    evals_result = xgb.evals_result()
    
    if model_name:
        os.makedirs(task_dir, exist_ok=True)
        xgb.save_model(f"{task_dir}/{model_name}.json")

    meta = {
        "num_class": num_class,
        "objective": objective,
        "eval_metric": eval_metric
    }

    return xgb, evals_result, meta

def evaluate_model(y_test, y_pred, model, x_test, focus_class=1):
    # - 二元分類：照原本邏輯（class 1 為正類）
    # - 多元分類：轉成 focus_class vs rest 的二元視角（輸出格式維持舊版，前端最省事）
    y_test = np.asarray(y_test).astype(int)
    x_test = np.asarray(x_test)

    # predict_proba for both binary / multiclass
    y_pred_proba = model.predict_proba(x_test)
    num_class = int(y_pred_proba.shape[1])

    # ===== 1) 產生「二元視角」y_true_bin / y_score =====
    if num_class == 2:
        # 二元：用原本的 class 1 當正類
        y_true_bin = y_test
        y_score = y_pred_proba[:, 1]
    else:
        # 多元：focus_class vs rest
        focus_class = int(focus_class)
        y_true_bin = (y_test == focus_class).astype(int)
        y_score = y_pred_proba[:, focus_class]

    # ===== 2) 0.5 預設門檻的二元預測（供基本混淆矩陣/指標）=====
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

    # ===== 3) recall 門檻掃描（維持舊行為）=====
    thresh_list = []
    accuracy_score_list = []
    precision_score_list = []
    recall_score_list = []
    f1_score_list = []
    specificity_score_list = []
    npv_score_list = []
    confusion_matrix_list = []

    # 二元：完全複製你舊版的掃法（用 class 0 機率）
    if num_class == 2:
        score_for_threshold = y_pred_proba[:, 0]  # 舊版是用 x[0]
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

    # 多元：focus_class vs rest（用 y_true_bin / y_score，你現在新設計）
    else:
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

    # ===== 挑 best threshold：跟舊版一樣（recall>=標準，取 f1 最大）=====
    for recall_standard in [80, 85, 90, 95]:
        # 注意：recall_score_list 是 0~1，所以要用 recall_standard/100
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

def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=2000):
    # - 二元分類：照舊產生 shap_plot + shap_importance
    # - 多元分類：使用 focus_class 那一類的 shap（one-vs-rest 的「那一類輸出」視角）
    # - 只輸出一張 summary plot，讓前端改動最小
    result = {}
    try:
        # 轉成 numpy，並做抽樣避免資料太大造成 SHAP 很慢/爆記憶體
        x_test = np.asarray(x_test, dtype=np.float32)
        n = x_test.shape[0]
        if n > max_samples:
            idx = np.random.RandomState(30).choice(n, size=max_samples, replace=False)
            x_bg = x_test[idx]     # background / sample for explainer
            x_plot = x_test[idx]   # 也用同一批畫圖，速度更穩
        else:
            x_bg = x_test
            x_plot = x_test

        # 推類別數
        proba = model.predict_proba(x_bg[:1])
        num_class = int(proba.shape[1])

        # 建立 explainer：Tree 模型通常會自動用 TreeExplainer（shap.Explainer 會幫你選）
        explainer = shap.Explainer(model, x_bg)
        shap_values = explainer(x_plot)

        values = shap_values.values
        data = shap_values.data  # x_plot 對應資料

        # 二元 vs 多元取值
        if values.ndim == 3:
            # (samples, features, classes)
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= values.shape[2]:
                # 若 focus_class 超界，退回用預測機率最高的類別
                focus_class = int(np.argmax(model.predict_proba(x_plot[:1])[0]))

            values_for_plot = values[:, :, focus_class]
            result["shap_focus_class"] = int(focus_class)
            result["shap_num_class"] = int(values.shape[2])
        else:
            # (samples, features)
            values_for_plot = values
            result["shap_focus_class"] = 1
            result["shap_num_class"] = int(num_class)

        # 平均重要度（全樣本平均 |shap|）
        shap_importance = np.abs(values_for_plot).mean(axis=0)
        result["shap_importance"] = {
            feature_names[i]: float(val) for i, val in enumerate(shap_importance)
        }

        # beeswarm summary plot（單張）
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
    # - 二元分類：照原本 LIME（class 1 視角）
    # - 多元分類：用 focus_class 這一類來解釋（labels=[focus_class]），只輸出一張圖
    # - 回傳 key 維持 lime_plot / lime_example_0，前端不用改
    result = {}
    try:
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).astype(int)

        # 類別數（從 predict_proba 最穩）
        proba0 = model.predict_proba(x_test[:1])[0]
        num_class = int(len(proba0))

        # 二元：維持 class_0 / class_1；多元：自動產生 class_0..class_{K-1}
        class_names = [f"class_{i}" for i in range(num_class)]

        # focus_class 合法化
        if num_class == 2:
            focus_class = 1  # 二元固定用 class 1（和你原本一致）
        else:
            focus_class = int(focus_class)
            if focus_class < 0 or focus_class >= num_class:
                # 超界就用該 sample 的預測類別
                focus_class = int(np.argmax(model.predict_proba(x_test[[sample_index]])[0]))

        # LIME explainer：training_labels 可以給 y_test（你原本就是這樣）
        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
        )

        # 解釋指定 sample、指定 focus_class
        lime_result = lime_explainer.explain_instance(
            x_test[sample_index],
            model.predict_proba,
            num_features=num_features,
            labels=[focus_class]
        )

        # 圖（只畫 focus_class 那張）
        fig = lime_result.as_pyplot_figure(label=focus_class)
        fig.set_size_inches(10, 6)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)

        # 文字解釋（同樣只取 focus_class）
        result["lime_example_0"] = lime_result.as_list(label=focus_class)

        # meta（前端可不接）
        result["lime_focus_class"] = int(focus_class)
        result["lime_num_class"] = int(num_class)
        result["lime_sample_index"] = int(sample_index)

    except Exception as e:
        result["lime_error"] = str(e)

    return result

def plot_loss(evals_result):
    # - binary: 使用 logloss
    # - multiclass: 使用 mlogloss
    # - 自動判斷，不需要前端/呼叫端改任何東西
    # 自動判斷使用哪個 loss key
    if 'logloss' in evals_result['validation_0']:
        loss_key = 'logloss'
        ylabel = 'Log Loss'
        title = 'Training vs Validation Log Loss'
    elif 'mlogloss' in evals_result['validation_0']:
        loss_key = 'mlogloss'
        ylabel = 'Multi-class Log Loss'
        title = 'Training vs Validation Multi-class Log Loss'
    else:
        # 保險：找第一個含 loss 的 key
        loss_key = list(evals_result['validation_0'].keys())[0]
        ylabel = loss_key
        title = f'Training vs Validation {loss_key}'

    loss_train = evals_result['validation_0'][loss_key]
    loss_val = evals_result['validation_1'][loss_key]
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
    # - binary: 使用 error -> accuracy = 1 - error
    # - multiclass: 使用 merror -> accuracy = 1 - merror
    # - 自動判斷，不需要前端/呼叫端改任何東西
    # 自動判斷使用哪個 error key
    if 'error' in evals_result['validation_0']:
        err_key = 'error'
        title = 'Training vs Validation Accuracy'
    elif 'merror' in evals_result['validation_0']:
        err_key = 'merror'
        title = 'Training vs Validation Accuracy (Multi-class)'
    else:
        # 保險：從 keys 裡挑最像 error 的
        keys0 = list(evals_result['validation_0'].keys())
        # 先找包含 'error' 的（最合理）
        candidates = [k for k in keys0 if 'error' in k.lower()]
        err_key = candidates[0] if candidates else keys0[0]
        title = f'Training vs Validation Accuracy ({err_key})'

    err_train = evals_result['validation_0'][err_key]
    err_val = evals_result['validation_1'][err_key]

    acc_train = [1 - e for e in err_train]
    acc_val = [1 - e for e in err_val]
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
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return image_base64

def kfold_evaluation(X, y, cv_folds, model_name, n_estimators, learning_rate, max_depth,
                     feature_names, task_dir, focus_class=1):
    # - 二元：正常 k-fold
    # - 多元：用 focus_class 做 one-vs-rest 的二元視角評估（前端改動最小）
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=30)
    folds_result = []

    # 用來算平均（沿用 evaluate_model 的輸出）
    acc_list, rec_list, prec_list, f1_list = [], [], [], []
    total_tn = total_fp = total_fn = total_tp = 0

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_fold_name = f"{model_name}_fold_{fold}"

        # train_xgb：你目前的版本回傳 (model, evals_result)
        model, evals_result, meta = train_xgb(
            X_train, y_train, X_test, y_test,
            model_fold_name, n_estimators, learning_rate, max_depth, task_dir
        )

        # ===== 評估：直接用你已經改好的 evaluate_model（會自動處理 binary/multiclass）=====
        # y_pred 其實 evaluate_model 不一定需要，但你原本 signature 有 y_pred，所以保留
        y_pred = model.predict(X_test)
        fold_eval = evaluate_model(y_test, y_pred, model, X_test, focus_class=focus_class)

        # fold_eval 裡面已經有：
        # - confusion_matrix: {tn,fp,fn,tp}
        # - metrics: {accuracy, recall, precision, f1_score}
        # - roc
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

        # ===== 圖：loss/accuracy（你也已經改成自動 logloss/mlogloss、error/merror）=====
        loss_base64 = plot_loss(evals_result)
        acc_base64 = plot_accuracy(evals_result)

        # ===== SHAP/LIME：也固定 focus_class，輸出一張圖（前端不改）=====
        shap_result = explain_with_shap(model, X_test, feature_names, focus_class=focus_class)
        lime_result = explain_with_lime(model, X_test, y_test, feature_names, focus_class=focus_class)

        folds_result.append({
            "fold": int(fold),

            # 直接把 evaluate_model 的核心結果塞進來（格式跟你原本一致）
            "metrics": fold_eval.get("metrics"),
            "confusion_matrix": fold_eval.get("confusion_matrix"),

            # 你原本每 fold 存的 roc / plots
            "roc": fold_eval.get("roc"),
            "loss_plot": loss_base64,
            "accuracy_plot": acc_base64,

            # SHAP
            "shap_plot": shap_result.get("shap_plot"),
            "shap_importance": shap_result.get("shap_importance"),
            "shap_error": shap_result.get("shap_error"),

            # LIME
            "lime_plot": lime_result.get("lime_plot"),
            "lime_example_0": lime_result.get("lime_example_0"),
            "lime_error": lime_result.get("lime_error"),

            # 額外資訊（不影響前端）
            "num_class": fold_eval.get("num_class"),
            "focus_class": fold_eval.get("focus_class"),

            # 保留你原本的 recall_80/85/90/95（如果你前端有顯示）
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

    result = {
        "status": "success",
        "num_class": int(folds_result[0].get("num_class", 2)) if folds_result else 2,
        "focus_class": int(focus_class),
        "folds": folds_result,
        "average": avg_result
    }
    return result

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth, task_dir):
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
            model, evals_result, meta = train_xgb(x_train, y_train, x_test, y_test, model_name, n_estimators, learning_rate, max_depth, task_dir)
            y_pred = model.predict(x_test)
            results = evaluate_model(y_test, y_pred, model, x_test, focus_class=1)  # focus_class 預設 1
            results["loss_plot"] = plot_loss(evals_result)
            results["accuracy_plot"] = plot_accuracy(evals_result)
            shap_result = explain_with_shap(model, x_test, feature_names, focus_class=1)  # focus_class 預設 1
            results.update(shap_result)
            lime_result = explain_with_lime(model, x_test, y_test, feature_names, focus_class=1)  # focus_class 預設 1
            results.update(lime_result)
        except ValueError as e:
            print(json.dumps({
                "status": "error",
                "message": f"{e}",
            }))
            return
        
    elif split_strategy == "k_fold":
        try:
            # 重新打包 train function，這樣就不用傳遞超參數
            results = kfold_evaluation(x, y, int(split_value), model_name, n_estimators, learning_rate, max_depth, feature_names, task_dir, focus_class=1)
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
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.n_estimators, args.learning_rate, args.max_depth, args.task_dir)
