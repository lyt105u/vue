# usage: python train_stacking.py upload/alice/長庚csv有答案.csv "[\"xgb\", \"lgbm\"]" label lgbm model/alice/20250721112233

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
from tool_train import prepare_data, NumpyEncoder, extract_base64_images_and_clean_json
import argparse

def save_oof_csv(meta_X, y, base_models, task_dir, filename='oof.csv'):
    os.makedirs(task_dir, exist_ok=True)
    df = pd.DataFrame(meta_X, columns=base_models)
    df['y'] = y
    path = os.path.join(task_dir, filename)
    df.to_csv(path, index=False)
    return


def main(file_path, base_models, label_column, meta_model, task_dir):
    results = {
        "status": "success",
        "base_results": {},
        "base_models": base_models,
        "meta_model": meta_model
    }
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
    meta_features = {name: np.zeros(X.shape[0]) for name in base_models}
    # base_models_per_fold = {name: [] for name in model_names}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        for name in base_models:
            module = import_module(f"stacking_{name}")
            preds = module.train_fold(
                X_train, y_train, X_val
            )
            meta_features[name][val_idx] = preds
    # 組成 meta_X，真實標籤用 y
    meta_X = np.column_stack([meta_features[name] for name in base_models])
    save_oof_csv(meta_X, y, base_models, task_dir)

    # 4. 訓練 meta model (OOF 階段)
    meta_module = import_module(f"stacking_{meta_model}")
    meta_results =  meta_module.retrain(meta_X, y, base_models, task_dir, '0.8', False, 'meta')
    results["meta_results"] = meta_results

    # 5. Retrain base models on full data
    for name in base_models:
        module = import_module(f"stacking_{name}")
        base_result = module.retrain(X, y, feature_names, task_dir, '0.8', True, 'base')
        results["base_results"][f"{name}"] = base_result

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

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
    parser.add_argument('base_models', type=json.loads) # JSON list of model names, e.g. \'["xgb","lgbm"]\'
    parser.add_argument("label_column", type=str)
    parser.add_argument("meta_model", type=str)
    parser.add_argument("task_dir", type=str)

    args = parser.parse_args()
    main(args.file_path, args.base_models, args.label_column, args.meta_model, args.task_dir)
