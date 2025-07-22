# usage: python train_stacking.py upload/高醫訓練xlsx.xlsx "[\"xgb\", \"lgbm\"]" label model/user1/20250721112233

import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import joblib
import os
import json
import sys
import os
import pandas as pd
from tool_train import prepare_data
import argparse

def save_oof_csv(meta_X, y, model_names, task_dir, filename='oof.csv'):
    os.makedirs(task_dir, exist_ok=True)
    df = pd.DataFrame(meta_X, columns=model_names)
    df['y'] = y
    path = os.path.join(task_dir, filename)
    df.to_csv(path, index=False)
    return


def main(file_path, model_names, label_column, task_dir):
    # 1. 讀取資料
    try:
        X, y, feature_names = prepare_data(file_path, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return
    os.makedirs(task_dir, exist_ok=True)

    # 2. K-Fold 切分
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(X))  # 確保所有 model 用相同 folds

    # 3. 產生 OOF meta features
    meta_features = {name: np.zeros(X.shape[0]) for name in model_names}
    base_models_per_fold = {name: [] for name in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # print(fold_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        for name in model_names:
            # print(name)
            module = import_module(f"stacking_{name}")
            preds = module.train_fold(
                X_train, y_train, X_val
            )
            meta_features[name][val_idx] = preds

    # 組成 meta_X，真實標籤用 y
    meta_X = np.column_stack([meta_features[name] for name in model_names])
    save_oof_csv(meta_X, y, model_names, task_dir)
    # print("OOF generated.")
    print(json.dumps({
        "status": "success",
        "message": "OOF generated.",
    }))

    # 4. 訓練 meta model (OOF 階段)
    # meta_model = LogisticRegression(**meta_params)
    # meta_model.fit(meta_X, y)
    # joblib.dump(meta_model, os.path.join(output_dir, 'meta_model_oof.pkl'))
    # joblib.dump(base_models_per_fold, os.path.join(output_dir, 'base_models_per_fold.pkl'))

    # # 5. Retrain base models on full data
    # for name in model_names:
    #     module = import_module(f"stacking_{name}")
    #     model_path = os.path.join(output_dir, f"{name}_final_model.pkl")
    #     xai_dir = os.path.join(output_dir, f"{name}_xai")
    #     os.makedirs(xai_dir, exist_ok=True)
    #     module.retrain(X, y, model_params[name], model_path, xai_dir)

    # # 6. Retrain meta model on full data
    # full_preds = []
    # for name in model_names:
    #     models = base_models_per_fold[name]
    #     # 平均所有 folds retrain 還原之預測 (此處使用 X 的所有樣本)
    #     fold_probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    #     avg_probs = fold_probs.mean(axis=1)
    #     full_preds.append(avg_probs)

    # full_meta_X = np.column_stack(full_preds)
    # meta_full_model = LogisticRegression(**meta_params)
    # meta_full_model.fit(full_meta_X, y)
    # joblib.dump(meta_full_model, os.path.join(output_dir, 'meta_model_full.pkl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument('model_names', type=json.loads) # JSON list of model names, e.g. \'["xgb","lgbm"]\'
    parser.add_argument("label_column", type=str)
    # parser.add_argument("split_value", type=str)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.model_names, args.label_column, args.task_dir)
