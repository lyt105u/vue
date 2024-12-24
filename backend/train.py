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
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "scikit-learn==1.5.2"],
    stdout=subprocess.DEVNULL
)
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
from catboost import CatBoostClassifier
import joblib
import json
import argparse

def train_and_save_model(model_type, data_path):
    # 讀取訓練資料
    train_data = pd.read_csv(data_path)

    x_train = train_data.drop('label', axis=1)
    y_train = train_data['label']

    # 檢查訓練集的類別分佈
    unique, counts_train = np.unique(y_train, return_counts=True)
    # print("Training set class distribution:", dict(zip(unique, counts_train)))

    # 根據輸入的模型類型訓練並儲存對應的模型
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
        xgb.save_model('model_xgb.json')
        print("success")

    elif model_type == 'random_forest':
        rf = RandomForestClassifier(
            n_estimators=900,  # 決策樹的數量
            max_depth=50,       # 最大深度
            random_state=0,    # 隨機種子
            n_jobs=-1          # 使用所有可用的 CPU 核心
        )
        rf.fit(x_train, y_train)
        joblib.dump(rf, 'model_rf.pkl')
        print("success")

    elif model_type == 'lightgbm':
        lightgbm = LGBMClassifier(verbose=-1)
        lightgbm.fit(x_train, y_train)
        joblib.dump(lightgbm, 'model_lightgbm.pkl')
        print("success")

    elif model_type == 'catboost':
        catboost = CatBoostClassifier(verbose=0)  # 禁用訓練輸出
        catboost.fit(x_train, y_train)
        joblib.dump(catboost, 'model_catboost.pkl')
        print("success")

    else:
         print("fail");
        
def main(model_file, test_file):
    results = train_and_save_model(model_file, test_file)

if __name__ == "__main__":
    # 使用 argparse 處理命令列參數
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('data_name', type=str)
    
    args = parser.parse_args()
    main(args.model_name, args.data_name)
