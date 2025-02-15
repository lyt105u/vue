import json
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import joblib
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, evaluate_model, kfold_evaluation

def train_lr(x_train, y_train, model_name):
    logistic_reg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver='lbfgs', C=0.1)
    )
    logistic_reg.fit(x_train, y_train)
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        joblib.dump(logistic_reg, f"model/{model_name}.pkl")

    return logistic_reg

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
        model = train_lr(x_train, y_train, model_name)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)
    elif split_strategy == "k_fold":
        results = kfold_evaluation(x, y, split_value, train_lr)
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
