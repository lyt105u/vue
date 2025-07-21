from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

def train_fold(X_train, y_train, X_val, penalty="l2", alpha=1.0, max_iter=500):
    logistic_reg = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss='log_loss',
            penalty=penalty,
            alpha=alpha,
            max_iter=max_iter,
            warm_start=True,
        )
    )
    logistic_reg.fit(X_train, y_train)
    val_preds = logistic_reg.predict_proba(X_val)[:, 1]
    return val_preds
