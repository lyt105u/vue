import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

def train_fold(X_train, y_train, X_val, batch_size=256, max_epochs=2, patience=10):
    tabnet = TabNetClassifier(verbose=0)    # verbose 隱藏輸出
    x_train_np = np.array(X_train, dtype=np.float32)    # 確保 x_train 和 y_train 是 numpy.ndarray，而非 object
    y_train_np = np.array(y_train, dtype=np.int64)
    x_val_np = np.array(X_val, dtype=np.float32)    # 確保 x_train 和 y_train 是 numpy.ndarray，而非 object

    tabnet.fit(
        x_train_np, y_train_np,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    val_preds = tabnet.predict_proba(x_val_np)[:, 1]
    return val_preds