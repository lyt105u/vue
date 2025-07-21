from sklearn.neural_network import MLPClassifier

def train_fold(X_train, y_train, X_val, hidden_layer_1=128, hidden_layer_2=64, hidden_layer_3=None, activation="relu", learning_rate_init=0.001, max_iter=300, n_iter_no_change=50):
    hidden_layer_sizes = tuple(filter(lambda x: x is not None, [hidden_layer_1, hidden_layer_2, hidden_layer_3]))
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        solver='adam',
        alpha=0.0001,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=n_iter_no_change,
        verbose=False
    )
    mlp.fit(X_train, y_train)
    val_preds = mlp.predict_proba(X_val)[:, 1]
    return val_preds
