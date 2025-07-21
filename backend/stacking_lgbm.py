import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from lightgbm import LGBMClassifier

def train_fold(X_train, y_train, X_val, n_estimators=100, learning_rate=0.1, max_depth=-1, num_leaves=31):
    # Train LGBMClassifier on a given fold and return validation predictions and the model.
    lightgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        verbose=-1
    )
    lightgbm.fit(X_train, y_train)
    val_preds = lightgbm.predict_proba(X_val)[:, 1]
    return val_preds


def retrain(X, y, params, output_model_path, output_xai_dir):
    # Retrain LGBMClassifier on full data, save model, and generate XAI (SHAP & LIME).
    model = LGBMClassifier(**params)
    model.fit(X, y)
    joblib.dump(model, output_model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    np.save(f"{output_xai_dir}/shap_values.npy", shap_values)

    explainer_lime = LimeTabularExplainer(
        training_data=X,
        feature_names=None,
        class_names=['0', '1'],
        mode='classification'
    )
    lime_explanations = [
        explainer_lime.explain_instance(X[i], model.predict_proba).as_list()
        for i in range(min(10, X.shape[0]))
    ]
    joblib.dump(lime_explanations, f"{output_xai_dir}/lime_explanations.pkl")

    return model, shap_values, lime_explanations