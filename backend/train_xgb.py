import json
import argparse
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, evaluate_model, kfold_evaluation

def train_xgb(x_train, y_train, model_name):
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
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        xgb.save_model(f"model/{model_name}.json")

    return xgb

def main(file_name, label_column, split_strategy, split_value, model_name):
    try:
        x, y = prepare_data(file_name, label_column)
    except ValueError as e:
        print(json.dumps({
            "status": "error",
            "message": f"{e}",
        }))
        return

    if split_strategy == "train_test_split":
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=float(split_value), stratify=y, random_state=30
        )
        model = train_xgb(x_train, y_train, model_name)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)
    elif split_strategy == "k_fold":
        results = kfold_evaluation(x, y, split_value, train_xgb)
    else:
        print(json.dumps({
            "status": "error",
            "message": "Unsupported split strategy"
        }))
        return

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("label_column", type=str)
    parser.add_argument("split_strategy", type=str)
    parser.add_argument("split_value", type=str)
    parser.add_argument("model_name", type=str)

    args = parser.parse_args()
    main(args.file_name, args.label_column, args.split_strategy, args.split_value, args.model_name)
