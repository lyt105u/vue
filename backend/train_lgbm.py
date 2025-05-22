import json
import argparse
import os
from lightgbm import LGBMClassifier
import joblib
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, evaluate_model, kfold_evaluation

def train_lgbm(x_train, y_train, model_name, n_estimators, learning_rate, max_depth, num_leaves):
    lightgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        verbose=-1
    )
    lightgbm.fit(x_train, y_train)
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        joblib.dump(lightgbm, f"model/{model_name}.pkl")

    return lightgbm

def main(file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth, num_leaves):
    try:
        x, y = prepare_data(file_path, label_column)
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
        model = train_lgbm(x_train, y_train, model_name, n_estimators, learning_rate, max_depth, num_leaves)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)
    elif split_strategy == "k_fold":
        # 重新打包 train function，這樣就不用傳遞超參數
        def train_lgbm_wrapped(x_train, y_train, model_name):
            lightgbm = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                verbose=-1
            )
            lightgbm.fit(x_train, y_train)
    
            if model_name:
                os.makedirs("model", exist_ok=True)
                joblib.dump(lightgbm, f"model/{model_name}.pkl")

            return lightgbm
        results = kfold_evaluation(x, y, split_value, train_lgbm_wrapped)
    else:
        print(json.dumps({
            "status": "error",
            "message": "Unsupported split strategy"
        }))
        return

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
    parser.add_argument("num_leaves", type=int)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.n_estimators, args.learning_rate, args.max_depth, args.num_leaves)
