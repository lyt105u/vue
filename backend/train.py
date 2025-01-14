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

def prepare_data(file_name, label_column, train_size):
    folder_path = "data"
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
    os.makedirs("model", exist_ok=True) # 確保 model 資料夾存在
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
        xgb.save_model(f"model/{model_name}.json")
        return xgb

    elif model_type == 'random_forest':
        rf = RandomForestClassifier(
            n_estimators=900,   # 決策樹的數量
            max_depth=50,       # 最大深度
            random_state=0,     # 隨機種子
            n_jobs=-1           # 使用所有可用的 CPU 核心
        )
        rf.fit(x_train, y_train)
        joblib.dump(rf, f"model/{model_name}.pkl")
        return rf

    elif model_type == 'lightgbm':
        lightgbm = LGBMClassifier(verbose=-1)
        lightgbm.fit(x_train, y_train)
        joblib.dump(lightgbm, f"model/{model_name}.pkl")
        return lightgbm

    else:
        print(json.dumps({
            "status": "error",
            "message": "Invalid model type",
        }))
        sys.exit(1)

def kfold_evaluation(X, y, model, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=30)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, zero_division=0))
        recall_list.append(recall_score(y_test, y_pred, zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, zero_division=0))

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='m', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle="-.", color='0')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # plt.savefig('roc.png')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return {
        "status": "success",
        "metrics": {
            "accuracy": np.mean(accuracy_list),
            "precision": np.mean(precision_list),
            "recall": np.mean(recall_list),
            "f1_score": np.mean(f1_list),
        },
        "roc": image_base64
    }
    
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
        high_recall_best_f1_score_index = np.argmax(high_recall_f1score)

        best_recall = high_recall_recall[high_recall_best_f1_score_index]
        best_specificity = high_recall_specificity[high_recall_best_f1_score_index]
        best_precision = high_recall_precision[high_recall_best_f1_score_index]
        best_npv = high_recall_npv[high_recall_best_f1_score_index]
        best_f1 = high_recall_f1score[high_recall_best_f1_score_index]
        best_f2 = high_recall_f2score[high_recall_best_f1_score_index]
        best_accuracy = high_recall_accuracy[high_recall_best_f1_score_index]
        best_confusion_matrix = high_recall_confusion_matrix[high_recall_best_f1_score_index]

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
        
def main(model_type, file_name, label_column, train_size, model_name):
    train_size = float(train_size)
    x, y = prepare_data(file_name, label_column, train_size)

    if train_size == 1:
        model = train_model(model_type, model_name, x, y)
        results = kfold_evaluation(x, y, model)

    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=float(train_size), shuffle=True, stratify=y, random_state=30)
        model = train_model(model_type, model_name, x_train, y_train)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help="Type of model to train (xgb, random_forest, lightgbm)")
    parser.add_argument('file_name', type=str, help="File name of the data")
    parser.add_argument('label_column', type=str, help="The column chosen to be label")
    parser.add_argument('train_size', type=str, help="The size of training set")
    parser.add_argument('model_name', type=str, help="The name of the trained model that stored in model\ ")

    args = parser.parse_args()
    main(args.model_type, args.file_name, args.label_column, args.train_size, args.model_name)
