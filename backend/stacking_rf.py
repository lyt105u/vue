from sklearn.ensemble import RandomForestClassifier

def train_fold(X_train, y_train, X_val, n_estimators=900, max_depth=50, random_state=0, n_jobs=-1):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,  # 決策樹的數量
        max_depth=max_depth,        # 最大深度
        random_state=random_state,  # 隨機種子
        n_jobs=n_jobs               # 使用所有可用的 CPU 核心
    )
    rf.fit(X_train, y_train)
    val_preds = rf.predict_proba(X_val)[:, 1]
    return val_preds
