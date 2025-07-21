import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBClassifier

def train_fold(X_train, y_train, X_val, n_estimators=100, learning_rate=0.300000012, max_depth=6):
    # Train XGBClassifier on a given fold and return validation predictions and the model.

    # Args:
    #   X_train: np.ndarray of training features for this fold
    #   y_train: np.ndarray of training labels for this fold
    #   X_val: np.ndarray of validation features for this fold
    #   params: dict of XGBClassifier parameters

    # Returns:
    #   val_preds: np.ndarray of predicted probabilities for class 1 on X_val
    #   model: trained XGBClassifier instance
    xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                        gamma=0, device='cuda', importance_type=None,
                        interaction_constraints='', learning_rate=learning_rate,
                        max_delta_step=0, max_depth=max_depth, min_child_weight=1,
                        monotone_constraints='()', n_estimators=n_estimators, n_jobs=72,
                        num_parallel_tree=1, random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                        validate_parameters=1, verbosity=None, eval_metric=['logloss', 'error'],
                        tree_method='gpu_hist',
                        predictor='gpu_predictor')
    xgb.fit(X_train, y_train)
    val_preds = xgb.predict_proba(X_val)[:, 1]
    return val_preds


# def retrain(X, y, task_dir, n_estimators, learning_rate, max_depth):
#     # Retrain XGBClassifier on full data, save model, and generate XAI (SHAP & LIME).
#     xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                         colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
#                         gamma=0, device='cuda', importance_type=None,
#                         interaction_constraints='', learning_rate=learning_rate,
#                         max_delta_step=0, max_depth=max_depth, min_child_weight=1,
#                         monotone_constraints='()', n_estimators=n_estimators, n_jobs=72,
#                         num_parallel_tree=1, random_state=0,
#                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#                         validate_parameters=1, verbosity=None, eval_metric=['logloss', 'error'],
#                         tree_method='gpu_hist',
#                         predictor='gpu_predictor')
#     xgb.fit(X, y)
#     joblib.dump(model, output_model_path)

#     # SHAP values
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     np.save(f"{output_xai_dir}/shap_values.npy", shap_values)

#     # LIME explanations (first 10 instances)
#     explainer_lime = LimeTabularExplainer(
#         training_data=X,
#         feature_names=None,
#         class_names=['0', '1'],
#         mode='classification'
#     )
#     lime_explanations = [
#         explainer_lime.explain_instance(X[i], model.predict_proba).as_list()
#         for i in range(min(10, X.shape[0]))
#     ]
#     joblib.dump(lime_explanations, f"{output_xai_dir}/lime_explanations.pkl")

#     return model, shap_values, lime_explanations
