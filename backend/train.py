import subprocess
import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-learn"])
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "lightgbm"],
#     stdout=subprocess.DEVNULL
# )
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "catboost"],
#     stdout=subprocess.DEVNULL
# )
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "scikit-learn==1.5.2"],
#     stdout=subprocess.DEVNULL
# )
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "--upgrade", "xgboost"],
#     stdout=subprocess.DEVNULL
# )
# scikit-learn 1.5.2
# pip 23.2.1

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
import joblib
import json
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import base64
import io
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def prepare_data(file_name, label_column):
    folder_path = "data/train"
    if not os.path.exists(folder_path):
        print(json.dumps({
            "status": "error",
            "message": f"Directory '{folder_path}' doesn't exist.",
        }))
        sys.exit(1)
    
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        print(json.dumps({
            "status": "error",
            "message": "Unsupported file format. Please provide a CSV or Excel file.",
        }))
        sys.exit(1)

    if label_column not in df.columns:
        print(json.dumps({
            "status": "error",
            "message": f"Label column '{label_column}' not found in the dataset.",
        }))
        sys.exit(1)
    
    x = df.drop(columns=[label_column]).values
    y = df[label_column].values.astype(float)

    return x, y

def train_model(model_type, model_name, x_train, y_train):
    # 若 model_name 為空，則不儲存
    os.makedirs("model", exist_ok=True) # 確保 model 資料夾存在

    field_count = x_train.shape[1]
    field_count_file = "model/field_counts.json"
    if os.path.exists(field_count_file):
        with open(field_count_file, "r") as f:
            field_counts = json.load(f)
    else:
        field_counts = {}

    if model_type == 'xgb':
        xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                            gamma=0, device='cpu', importance_type=None,
                            interaction_constraints='', learning_rate=0.300000012,
                            max_delta_step=0, max_depth=6, min_child_weight=1,
                            monotone_constraints='()', n_estimators=100, n_jobs=72,
                            num_parallel_tree=1, random_state=0,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                            tree_method='exact', validate_parameters=1, verbosity=None)
        xgb.fit(x_train, y_train)
        if model_name != '':
            xgb.save_model(f"model/{model_name}.json")
            field_counts[model_name + ".json"] = field_count
            with open(field_count_file, "w") as f:
                json.dump(field_counts, f, indent=4)
        return xgb

    elif model_type == 'random_forest':
        rf = RandomForestClassifier(
            n_estimators=900,   # 決策樹的數量
            max_depth=50,       # 最大深度
            random_state=0,     # 隨機種子
            n_jobs=-1           # 使用所有可用的 CPU 核心
        )
        rf.fit(x_train, y_train)
        if model_name != '':
            joblib.dump(rf, f"model/{model_name}.pkl")
            field_counts[model_name + ".pkl"] = field_count
            with open(field_count_file, "w") as f:
                json.dump(field_counts, f, indent=4)
        return rf

    elif model_type == 'lightgbm':
        lightgbm = LGBMClassifier(verbose=-1)
        lightgbm.fit(x_train, y_train)
        if model_name != '':
            joblib.dump(lightgbm, f"model/{model_name}.pkl")
            field_counts[model_name + ".pkl"] = field_count
            with open(field_count_file, "w") as f:
                json.dump(field_counts, f, indent=4)
        return lightgbm
    
    elif model_type == 'logistic_regression':
        logistic_reg = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, solver='lbfgs', C=0.1)
        )
        logistic_reg.fit(x_train, y_train)
        if model_name != '':
            joblib.dump(logistic_reg, f"model/{model_name}.pkl")
            field_counts[model_name + ".pkl"] = field_count
            with open(field_count_file, "w") as f:
                json.dump(field_counts, f, indent=4)
        return logistic_reg

    else:
        print(json.dumps({
            "status": "error",
            "message": "Invalid model type",
        }))
        sys.exit(1)

def kfold_evaluation(X, y, model_type, split_value):
    cv_folds = int(split_value)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    # 初始化累加变量
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    tpr_list = []
    fpr_list = []
    auc_list = []
    y_test_all = []
    y_pred_proba_all = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train model on the current fold
        model = train_model(model_type, '', X_train, y_train)

        # Predict probabilities for ROC curve and labels for confusion matrix
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        auc_list.append(auc_score)

        # Collect all true labels and predicted probabilities for aggregated ROC
        y_test_all.extend(y_test)
        y_pred_proba_all.extend(y_pred_proba)

    # Average metrics
    avg_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    avg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_auc = np.mean(auc_list)

    # Plot ROC Curve
    fpr_agg, tpr_agg, _ = roc_curve(y_test_all, y_pred_proba_all)
    plt.figure()
    plt.plot(fpr_agg, tpr_agg, color='m', label=f"ROC curve (AUC = {avg_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='0', linestyle="--")  # Baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    # Save ROC Curve as Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    roc_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Return results
    result = {
        "status": "success",
        "metrics": {
            "accuracy": avg_accuracy * 100,
            "recall": avg_recall * 100,
            "precision": avg_precision * 100,
            "f1_score": avg_f1 * 100,
            "auc": avg_auc * 100
        },
        "confusion_matrix": {
            "true_negative": total_tn,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "true_positive": total_tp,
        },
        "roc": roc_base64
    }
    return result
    
def evaluate_model(y_test, y_pred, model, x_test):
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    result = {
        "status": "success",
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
        
def main(model_type, file_name, label_column, split_strategy, split_value, model_name):
    x, y = prepare_data(file_name, label_column)

    if split_strategy == "k_fold":
        results = kfold_evaluation(x, y, model_type, split_value)
        model = train_model(model_type, model_name, x, y)

    elif split_strategy == "train_test_split":
        split_value = float(split_value)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_value, shuffle=True, stratify=y, random_state=30)
        model = train_model(model_type, model_name, x_train, y_train)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help="Type of model to train (xgb, random_forest, lightgbm)")
    parser.add_argument('file_name', type=str, help="File name of the data")
    parser.add_argument('label_column', type=str, help="The column chosen to be label")
    parser.add_argument('split_strategy', type=str, help="train_test_split or k_fold")
    parser.add_argument('split_value', type=str, help="The train_size or cv_folds depends on split_strategy")
    parser.add_argument('model_name', type=str, help="The name of the trained model that stored in directory model")

    args = parser.parse_args()
    main(args.model_type, args.file_name, args.label_column, args.split_strategy, args.split_value, args.model_name)
