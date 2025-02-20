import json
import argparse
import os
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, evaluate_model, kfold_evaluation

def train_mlp(x_train, y_train, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter):
    hidden_layer_sizes = tuple(filter(lambda x: x is not None, [hidden_layer_1, hidden_layer_2, hidden_layer_3]))
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        solver='adam',
        alpha=0.0001,
        random_state=0
    )
    mlp.fit(x_train, y_train)
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        joblib.dump(mlp, f"model/{model_name}.pkl")

    return mlp

def main(file_name, label_column, split_strategy, split_value, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter):
    # 處理空白的 layer
    def convert_arg(value):
        return None if value.lower() in ["null", "none", ""] else int(value)
    hidden_layer_1 = convert_arg(args.hidden_layer_1)
    hidden_layer_2 = convert_arg(args.hidden_layer_2)
    hidden_layer_3 = convert_arg(args.hidden_layer_3)

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
        model = train_mlp(x_train, y_train, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter)
        y_pred = model.predict(x_test)
        results = evaluate_model(y_test, y_pred, model, x_test)
    elif split_strategy == "k_fold":
        def train_mlp_wrapped(x_train, y_train, model_name):
            hidden_layer_sizes = tuple(filter(lambda x: x is not None, [hidden_layer_1, hidden_layer_2, hidden_layer_3]))
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter
            )
            mlp.fit(x_train, y_train)
            if model_name:
                os.makedirs("model", exist_ok=True)
                joblib.dump(mlp, f"model/{model_name}.pkl")
            return mlp

        results = kfold_evaluation(x, y, split_value, train_mlp_wrapped)
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
    parser.add_argument("hidden_layer_1", type=str)
    parser.add_argument("hidden_layer_2", type=str)
    parser.add_argument("hidden_layer_3", type=str)
    parser.add_argument("activation", type=str)
    parser.add_argument("learning_rate_init", type=float)
    parser.add_argument("max_iter", type=int)

    args = parser.parse_args()
    main(args.file_name, args.label_column, args.split_strategy, args.split_value, args.model_name, args.hidden_layer_1, args.hidden_layer_2, args.hidden_layer_3, args.activation, args.learning_rate_init, args.max_iter)
