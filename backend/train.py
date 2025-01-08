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

def prepare_data(data1, data0):
    pd.options.display.max_columns = None

    # 列印初始資訊
    # print('川崎症病人(原): ', len(data1))
    # print('川崎症病人(<=5歲): ', len(data1[data1['年齡(日)'] <= 2191]))
    # print('川崎症病人(<5歲): ', len(data1[data1['年齡(日)'] < 2191]))
    # print('發燒病人(原): ', len(data0))
    # print('發燒病人(<=5歲): ', len(data0[data0['年齡(日)'] <= 2191]))
    # print('發燒病人(<5歲): ', len(data0[data0['年齡(日)'] < 2191]))

    # 篩選年齡條件
    data1 = data1[data1['年齡(日)'] <= 1826]
    data0 = data0[data0['年齡(日)'] <= 1826]

    # 添加月份資訊
    data1['輸入日期(月)'] = data1['輸入日期'].dt.month
    data0['輸入日期(月)'] = data0['輸入日期'].dt.month

    # 獲取月份的 One-Hot 編碼
    data1_month = pd.get_dummies(data1['輸入日期(月)'], prefix='Month')
    data0_month = pd.get_dummies(data0['輸入日期(月)'], prefix='Month')

    # 合併數據
    data1 = pd.concat([data1, data1_month], axis=1).drop(columns=['輸入日期(月)', '輸入日期'])
    data0 = pd.concat([data0, data0_month], axis=1).drop(columns=['輸入日期(月)', '輸入日期'])

    # 打亂數據
    data1 = shuffle(data1, random_state=30)
    data0 = shuffle(data0, random_state=30)

    # 添加標籤
    data1['label'] = 1
    data0['label'] = 0

    # 合併所有數據
    all_train_data = pd.concat([data1, data0], ignore_index=True)

    # 分割特徵和標籤
    x = all_train_data.values[:, :-1]
    y = all_train_data.values[:, -1].astype(float)

    # 切分訓練集和測試集
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, stratify=y, random_state=30)

    return x_train, x_test, y_train, y_test

def train_model(model_type, x_train, y_train):
    os.makedirs("model", exist_ok=True) # 確保 model 資料夾存在
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
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
        xgb.save_model(f"model/xgb_{current_time}.json")
        return xgb

    elif model_type == 'random_forest':
        rf = RandomForestClassifier(
            n_estimators=900,  # 決策樹的數量
            max_depth=50,       # 最大深度
            random_state=0,    # 隨機種子
            n_jobs=-1          # 使用所有可用的 CPU 核心
        )
        rf.fit(x_train, y_train)
        joblib.dump(rf, f"model/rf_{current_time}.pkl")
        return rf

    elif model_type == 'lightgbm':
        lightgbm = LGBMClassifier(verbose=-1)
        lightgbm.fit(x_train, y_train)
        joblib.dump(lightgbm, f"model/lightgbm_{current_time}.pkl")
        return lightgbm

    else:
        raise ValueError("Invalid model type")
    
def evaluate_model(y_test, y_pred, model, x_test):
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
        
def main(model_type, data1_path, data0_path):
    data1 = pd.read_excel(os.path.join("data", data1_path))
    data0 = pd.read_excel(os.path.join("data", data0_path))

    x_train, x_test, y_train, y_test = prepare_data(data1, data0)

    model = train_model(model_type, x_train, y_train)

    y_pred = model.predict(x_test)
    # 確保型態一致
    y_test = y_test.astype(float)
    y_pred = y_pred.astype(float)

    results = evaluate_model(y_test, y_pred, model, x_test)
    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, help="Type of model to train (xgb, random_forest, lightgbm)")
    parser.add_argument('data1_path', type=str, help="Path to data labeled 1")
    parser.add_argument('data0_path', type=str, help="Path to data labeled 0")

    args = parser.parse_args()
    main(args.model_type, args.data1_path, args.data0_path)
