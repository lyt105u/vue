import json
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import joblib
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, evaluate_model, kfold_evaluation

def train_lr(x_train, y_train, model_name, penalty, C, solver, max_iter):
    logistic_reg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
        )
    )
    logistic_reg.fit(x_train, y_train)
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        joblib.dump(logistic_reg, f"model/{model_name}.pkl")

    return logistic_reg

def main(file_path, label_column, split_strategy, split_value, model_name, penalty, C, solver, max_iter):
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
        model = train_lr(x_train, y_train, model_name, penalty, C, solver, max_iter)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)
    elif split_strategy == "k_fold":
        def train_lr_wrapped(x_train, y_train, model_name):
            logistic_reg = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                )
            )
            logistic_reg.fit(x_train, y_train)
            
            if model_name:
                os.makedirs("model", exist_ok=True)
                joblib.dump(logistic_reg, f"model/{model_name}.pkl")

            return logistic_reg
        results = kfold_evaluation(x, y, split_value, train_lr_wrapped)
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
    parser.add_argument("penalty", type=str)
    parser.add_argument("C", type=float)
    parser.add_argument("solver", type=str)
    parser.add_argument("max_iter", type=int)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.penalty, args.C, args.solver, args.max_iter)
