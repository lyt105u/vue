import json
import argparse
import os
import numpy as np

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet==4.1.0"])
    from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.model_selection import train_test_split
from tool_train import prepare_data, NumpyEncoder, evaluate_model, kfold_evaluation
from contextlib import contextmanager
import sys

@contextmanager
def suppress_stdout():
    # 暫時關閉標準輸出來隱藏 `save_model()` 的訊息
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def train_tabnet(x_train, y_train, model_name, batch_size, max_epochs, patience):
    tabnet = TabNetClassifier(verbose=0)    # verbose 隱藏輸出
    x_train_np = np.array(x_train, dtype=np.float32)    # 確保 x_train 和 y_train 是 numpy.ndarray，而非 object
    y_train_np = np.array(y_train, dtype=np.int64)

    tabnet.fit(
        x_train_np, y_train_np,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    if model_name:
        os.makedirs("model", exist_ok=True)
        # 使用 suppress_stdout() 來隱藏 `save_model()` 的輸出
        with suppress_stdout():
            tabnet.save_model(f"model/{model_name}")

    return tabnet

def main(file_path, label_column, split_strategy, split_value, model_name, batch_size, max_epochs, patience):
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
        model = train_tabnet(x_train, y_train, model_name, batch_size, max_epochs, patience)
        y_pred = model.predict(np.array(x_test, dtype=np.float32))  # 確保 x_test 在傳入前轉換為 numpy.float32
        results = evaluate_model(y_test, y_pred, model, np.array(x_test, dtype=np.float32))
    elif split_strategy == "k_fold":
        def train_tabnet_wrapped(x_train, y_train, model_name):
            tabnet = TabNetClassifier(verbose=0)    # verbose 隱藏輸出
            x_train_np = np.array(x_train, dtype=np.float32)    # 確保 x_train 和 y_train 是 numpy.ndarray，而非 object
            y_train_np = np.array(y_train, dtype=np.int64)

            tabnet.fit(
                x_train_np, y_train_np,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
            
            if model_name:
                os.makedirs("model", exist_ok=True)
                # 使用 suppress_stdout() 來隱藏 `save_model()` 的輸出
                with suppress_stdout():
                    tabnet.save_model(f"model/{model_name}")
            return tabnet
        
        results = kfold_evaluation(np.array(x, dtype=np.float32), y, split_value, train_tabnet_wrapped)
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
    parser.add_argument("batch_size", type=int)
    parser.add_argument("max_epochs", type=int)
    parser.add_argument("patience", type=int)

    args = parser.parse_args()
    main(args.file_path, args.label_column, args.split_strategy, args.split_value, args.model_name, args.batch_size, args.max_epochs, args.patience)
