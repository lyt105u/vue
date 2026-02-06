# usage:
# python predict.py model.json file <task_dir> --data_path input.xlsx --output_name result --label_column label --pred_column pred
# python predict.py model.json input <task_dir> --input_values 1.0 TRUE 3.0 FALSE
# read_csv 和 read_excel 會自行轉換數值型 (int 或 float)和布林型 (bool)

import argparse
from tool_train import NumpyEncoder, extract_base64_images_and_clean_json
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os
import json
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import sys
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_prop = font_manager.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import zipfile
import tempfile
import shutil


# =========================
# Helpers
# =========================
def _align_feature_names(feature_names, n_feat):
    if feature_names is None:
        return [f"f{i}" for i in range(n_feat)]
    if len(feature_names) == n_feat:
        return list(feature_names)
    if len(feature_names) > n_feat:
        return list(feature_names)[:n_feat]
    return list(feature_names) + [f"f{i}" for i in range(len(feature_names), n_feat)]

def _num_class_from_proba(proba_1row):
    return int(len(proba_1row))

def _pick_focus_class(num_class, focus_class):
    if num_class <= 2:
        return 1
    fc = int(focus_class)
    if fc < 0 or fc >= num_class:
        fc = 0
    return fc

def _is_catboost_model(m):
    return "catboost" in str(type(m)).lower()

def _is_lgbm_model(m):
    name = m.__class__.__name__.lower()
    return name.startswith("lgbm") or ("lightgbm" in str(type(m)).lower())

def _is_tabnet_model(m):
    return m.__class__.__name__.lower().startswith("tabnet")

def _predict_proba_any(model, X, model_path=None):
    # TabNet
    if _is_tabnet_model(model) or (model_path and str(model_path).lower().endswith(".zip")):
        Xn = np.asarray(X, dtype=np.float32)
        return model.predict_proba(Xn)

    Xn = np.asarray(X)

    # sklearn / xgb / catboost / lgbm / etc.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xn)

    # fallback: decision_function -> proba
    if hasattr(model, "decision_function"):
        s = model.decision_function(Xn)
        s = np.asarray(s)
        if s.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-s))
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        else:
            e = np.exp(s - np.max(s, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)

    raise ValueError("Model has no predict_proba / decision_function; cannot compute probabilities.")

def load_model(model_path):
    if model_path.lower().endswith(".json"):
        model = XGBClassifier()
        model.load_model(model_path)
    elif model_path.lower().endswith(".pkl"):
        model = joblib.load(model_path)
    elif model_path.lower().endswith(".zip"):
        model = TabNetClassifier()
        model.load_model(model_path)
    else:
        print(json.dumps({"status": "error", "message": f"Unsupported model format: {model_path}"}))
        sys.exit(1)
    return model

def load_data(data_path):
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        return pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


# =========================
# Explain: SHAP (binary/multiclass)
# =========================
def explain_with_shap(model, x_test, feature_names, focus_class=1, max_samples=2000, nsamples=100, model_path=None):
    result = {}
    try:
        x_test = np.asarray(x_test)
        if x_test.ndim != 2:
            raise ValueError(f"x_test must be 2D, got {x_test.shape}")

        n, n_feat = x_test.shape
        if n == 0:
            raise ValueError("x_test is empty")

        feature_names = _align_feature_names(feature_names, n_feat)

        if n > max_samples:
            idx = np.random.RandomState(30).choice(n, size=max_samples, replace=False)
            x_plot = x_test[idx]
        else:
            x_plot = x_test

        proba0 = _predict_proba_any(model, x_plot[:1], model_path=model_path)[0]
        num_class = _num_class_from_proba(proba0)
        fc = _pick_focus_class(num_class, focus_class)
        result["shap_num_class"] = int(num_class)
        result["shap_focus_class"] = int(fc)

        # ---------- CatBoost ----------
        if _is_catboost_model(model):
            from catboost import Pool
            x_cb = np.asarray(x_plot, dtype=np.float32)
            pool = Pool(x_cb, feature_names=feature_names)

            shap_raw = model.get_feature_importance(pool, fstr_type="ShapValues")
            shap_raw_arr = np.asarray(shap_raw)

            if isinstance(shap_raw, list):
                if fc < 0 or fc >= len(shap_raw):
                    fc = int(np.argmax(proba0))
                    result["shap_focus_class"] = int(fc)
                shap_vals = np.asarray(shap_raw[fc])
            else:
                if shap_raw_arr.ndim == 2:
                    shap_vals = shap_raw_arr
                elif shap_raw_arr.ndim == 3:
                    if fc < 0 or fc >= shap_raw_arr.shape[1]:
                        fc = int(np.argmax(proba0))
                        result["shap_focus_class"] = int(fc)
                    shap_vals = shap_raw_arr[:, fc, :]
                else:
                    raise ValueError(f"Unexpected CatBoost shap_raw shape={shap_raw_arr.shape}")

            shap_vals = np.asarray(shap_vals)
            if shap_vals.ndim != 2:
                raise ValueError(f"Unexpected CatBoost shap_vals shape={shap_vals.shape}")

            if shap_vals.shape[1] == n_feat + 1:
                shap_matrix = shap_vals[:, :n_feat]
            elif shap_vals.shape[1] == n_feat:
                shap_matrix = shap_vals
            else:
                raise ValueError(f"CatBoost SHAP dim mismatch: x={n_feat}, shap={shap_vals.shape[1]}")

            shap_importance = np.abs(shap_matrix).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, x_cb, feature_names=feature_names, show=False, max_display=min(30, n_feat))
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

        # ---------- LightGBM ----------
        if _is_lgbm_model(model):
            x_lgb = np.asarray(x_plot, dtype=np.float32)

            booster = None
            if hasattr(model, "booster_"):
                booster = model.booster_
            elif hasattr(model, "_Booster"):
                booster = model._Booster

            if booster is None:
                bg = x_lgb[: min(50, x_lgb.shape[0])]
                explainer = shap.KernelExplainer(lambda X: _predict_proba_any(model, X, model_path=model_path), bg)
                shap_values = explainer.shap_values(x_lgb[: min(50, x_lgb.shape[0])], nsamples=nsamples)
                x_used = x_lgb[: min(50, x_lgb.shape[0])]
            else:
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(x_lgb)
                x_used = x_lgb

            if isinstance(shap_values, list):
                shap_matrix = np.asarray(shap_values[fc]) if len(shap_values) > 1 else np.asarray(shap_values[0])
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_matrix = shap_values[:, fc, :] if shap_values.shape[1] == num_class else shap_values[:, :, fc]
            else:
                shap_matrix = np.asarray(shap_values)

            if shap_matrix.ndim != 2:
                raise ValueError(f"Unexpected LightGBM shap_matrix shape={shap_matrix.shape}")

            if shap_matrix.shape[1] == n_feat + 1:
                shap_matrix = shap_matrix[:, :n_feat]
            elif shap_matrix.shape[1] != n_feat:
                raise ValueError(f"LightGBM SHAP dim mismatch: x={n_feat}, shap={shap_matrix.shape[1]}")

            shap_importance = np.abs(shap_matrix).mean(axis=0)
            result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, x_used, feature_names=feature_names, show=False)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()
            return result

        # ---------- KernelExplainer fallback ----------
        x_kernel = np.asarray(x_plot, dtype=float)
        bg = x_kernel[: min(50, x_kernel.shape[0])]

        pred_fn = lambda X: _predict_proba_any(model, np.asarray(X, dtype=float), model_path=model_path)
        explainer = shap.KernelExplainer(pred_fn, bg)
        shap_values = explainer.shap_values(x_kernel[: min(50, x_kernel.shape[0])], nsamples=nsamples)
        x_used = x_kernel[: min(50, x_kernel.shape[0])]

        if isinstance(shap_values, list):
            shap_matrix = np.asarray(shap_values[fc]) if len(shap_values) > 1 else np.asarray(shap_values[0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, fc]
        else:
            shap_matrix = np.asarray(shap_values)

        if shap_matrix.ndim != 2:
            raise ValueError(f"Unexpected SHAP matrix shape={shap_matrix.shape}")

        if shap_matrix.shape[1] != n_feat:
            if shap_matrix.shape[1] == n_feat + 1:
                shap_matrix = shap_matrix[:, :n_feat]
            else:
                raise ValueError(f"SHAP dim mismatch: x={n_feat}, shap={shap_matrix.shape[1]}")

        shap_importance = np.abs(shap_matrix).mean(axis=0)
        result["shap_importance"] = {feature_names[i]: float(shap_importance[i]) for i in range(n_feat)}

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, x_used, feature_names=feature_names, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        result["shap_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()
        return result

    except Exception as e:
        result["shap_error"] = str(e)
        return result


# =========================
# Explain: LIME (binary/multiclass)
# =========================
def explain_with_lime(model, x_test, y_test_like, feature_names,
                      focus_class=1, sample_index=0, num_features=10, model_path=None):
    """
    通用 LIME（支援二元/多元）：
    - 自動偵測 num_class
    - binary: focus=1
    - multiclass: focus=focus_class（越界就改用 argmax(proba0)）
    - explain_instance 明確指定 labels=[focus]
    - as_pyplot_figure / as_list 也明確指定 label
    - ✅ 使用 _predict_proba_any 統一處理（TabNet / decision_function 也可）
    """
    result = {}
    try:
        x_test = np.asarray(x_test)
        if x_test.ndim != 2:
            raise ValueError(f"x_test must be 2D, got shape={x_test.shape}")
        n, n_feat = x_test.shape
        if n == 0:
            raise ValueError("x_test is empty")

        feature_names = _align_feature_names(feature_names, n_feat)

        y_test_like = np.asarray(y_test_like).astype(int).reshape(-1)
        if y_test_like.shape[0] != n:
            y_test_like = np.zeros(n, dtype=int)

        proba0 = _predict_proba_any(model, x_test[:1], model_path=model_path)[0]
        num_class = int(len(proba0))
        class_names = [f"class_{i}" for i in range(num_class)]
        result["lime_num_class"] = int(num_class)

        if num_class == 2:
            fc = 1
        else:
            fc = int(focus_class)
            if fc < 0 or fc >= num_class:
                fc = int(np.argmax(proba0))
        result["lime_focus_class"] = int(fc)
        result["lime_sample_index"] = int(sample_index)

        lime_explainer = LimeTabularExplainer(
            training_data=x_test,
            mode="classification",
            training_labels=y_test_like,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
        )

        pred_fn = lambda X: _predict_proba_any(model, np.asarray(X), model_path=model_path)

        exp = lime_explainer.explain_instance(
            x_test[sample_index],
            pred_fn,
            num_features=min(num_features, n_feat),
            labels=[fc],
        )

        fig = exp.as_pyplot_figure(label=fc)
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        result["lime_plot"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)

        result["lime_example_0"] = exp.as_list(label=fc)

    except Exception as e:
        result["lime_error"] = str(e)

    return result


# =========================
# Prediction (single model)
# =========================
def predict_labels(model, model_path, data, label_column, pred_column, focus_class=1, return_proba=True):
    """
    - 如果 data 裡有 label_column：會 drop 掉再預測（不影響原資料欄位保留）
    - 二元：寫入 proba_class1
    - 多元：寫入 proba_class{focus} + proba_top
    """
    X_df = data.drop(columns=[label_column], errors="ignore").copy()
    feature_names = X_df.columns.tolist()

    if model_path.lower().endswith(".zip"):
        X = X_df.to_numpy(dtype=np.float32)
    else:
        X = X_df.values

    y_pred = model.predict(X)
    data[pred_column] = y_pred

    try:
        proba = _predict_proba_any(model, X, model_path=model_path)
        num_class = int(proba.shape[1])
        fc = _pick_focus_class(num_class, focus_class)

        if return_proba:
            if num_class == 2:
                data["proba_class1"] = proba[:, 1]
            else:
                data[f"proba_class{fc}"] = proba[:, fc]
                data["proba_top"] = np.max(proba, axis=1)
    except Exception:
        pass

    # ✅ 分開 try：SHAP 爆掉也不會影響 LIME
    explain_result = {}

    explain_result["shap"] = explain_with_shap(
        model, X, feature_names, focus_class=focus_class, model_path=model_path
    )

    explain_result["lime"] = explain_with_lime(
        model, X, y_pred, feature_names, focus_class=focus_class, model_path=model_path
    )

    return data, explain_result


def save_predictions(data, data_path, output_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)
    if data_path.endswith(".csv"):
        output_path = os.path.join(task_dir, f"{output_name}.csv")
        data.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif data_path.endswith(".xlsx"):
        output_path = os.path.join(task_dir, f"{output_name}.xlsx")
        data.to_excel(output_path, index=False)
    else:
        raise ValueError("Unsupported output format.")
    return output_path


def predict_input(model, model_path, input_values, focus_class=1):
    parsed_values = []
    for value in input_values:
        if isinstance(value, str):
            if value.upper() == "TRUE":
                parsed_values.append(1)
            elif value.upper() == "FALSE":
                parsed_values.append(0)
            else:
                parsed_values.append(float(value))
        else:
            parsed_values.append(float(value))

    X = np.array([parsed_values], dtype=np.float32) if model_path.lower().endswith(".zip") else np.array([parsed_values], dtype=float)

    pred = model.predict(X)
    pred_list = pred.tolist() if isinstance(pred, np.ndarray) else pred

    out = {"status": "success", "prediction": pred_list}

    try:
        proba = _predict_proba_any(model, X, model_path=model_path)[0]
        num_class = int(len(proba))
        fc = _pick_focus_class(num_class, focus_class)
        out["num_class"] = int(num_class)
        out["focus_class"] = int(fc)
        out["proba"] = proba.tolist()
        out["proba_focus"] = float(proba[fc])
        out["proba_top"] = float(np.max(proba))
    except Exception:
        pass

    return out


def predict_single_model(model_path, mode, task_dir, data_path=None, output_name=None, label_column="label",
                        pred_column="prediction", input_values=None, focus_class=1):
    model = load_model(model_path)
    try:
        if mode == "file":
            data = load_data(data_path)
            data_with_predictions, explanations = predict_labels(
                model, model_path, data, label_column, pred_column, focus_class=focus_class
            )
            save_predictions(data_with_predictions, data_path, output_name, task_dir)

            result = {"status": "success"}

            shap_result = explanations.get("shap", {})
            if shap_result.get("shap_plot"):
                result["shap_plot"] = shap_result["shap_plot"]
            if shap_result.get("shap_importance"):
                result["shap_importance"] = shap_result["shap_importance"]
            if shap_result.get("shap_error"):
                result["shap_error"] = shap_result["shap_error"]

            lime_result = explanations.get("lime", {})
            if lime_result.get("lime_plot"):
                result["lime_plot"] = lime_result["lime_plot"]
            if lime_result.get("lime_example_0"):
                result["lime_example_0"] = lime_result["lime_example_0"]
            if lime_result.get("lime_error"):
                result["lime_error"] = lime_result["lime_error"]

            return result

        else:
            return predict_input(model, model_path, input_values, focus_class=focus_class)

    except Exception as e:
        return {"status": "error", "message": f"{e}"}


# =========================
# Prediction (stacking)
# =========================
def _model_file_for_base(name):
    if name == "xgb":
        return f"base_{name}.json"
    if name == "tabnet":
        return f"base_{name}.zip"
    return f"base_{name}.pkl"

def _model_file_for_meta(name):
    if name == "xgb":
        return f"meta_{name}.json"
    if name == "tabnet":
        return f"meta_{name}.zip"
    return f"meta_{name}.pkl"

def _get_meta_feature_for_model(m, X, model_path, focus_class=1):
    proba = _predict_proba_any(m, X, model_path=model_path)
    num_class = int(proba.shape[1])
    fc = _pick_focus_class(num_class, focus_class)
    return proba[:, fc]

def predict_stacking(model_path, mode, task_dir, data_path=None, output_name=None, label_column="label",
                     pred_column="prediction", input_values=None, focus_class=1):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        config_path = os.path.join(temp_dir, "stacking_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        base_names = config.get("base_models", [])
        meta_name = config.get("meta_model")

        base_models = []
        base_model_paths = []
        for bn in base_names:
            fp = os.path.join(temp_dir, _model_file_for_base(bn))
            base_models.append(load_model(fp))
            base_model_paths.append(fp)

        meta_model_path = os.path.join(temp_dir, _model_file_for_meta(meta_name))
        meta_model = load_model(meta_model_path)

        if mode == "file":
            data = load_data(data_path)
            X_df = data.drop(columns=[label_column], errors="ignore").copy()

            base_cols = []
            for m, mp in zip(base_models, base_model_paths):
                X_in = X_df.to_numpy(dtype=np.float32) if str(mp).lower().endswith(".zip") else X_df.values
                base_cols.append(_get_meta_feature_for_model(m, X_in, mp, focus_class=focus_class))

            X_meta = np.column_stack(base_cols)
            meta_feature_df = pd.DataFrame(X_meta, columns=base_names)
            save_predictions(meta_feature_df, data_path, "meta_feature", task_dir)

            y_pred = meta_model.predict(X_meta)
            data[pred_column] = y_pred
            save_predictions(data, data_path, output_name, task_dir)

            explanations = {}
            explanations["shap"] = explain_with_shap(
                meta_model, X_meta, base_names, focus_class=focus_class, model_path=meta_model_path
            )
            explanations["lime"] = explain_with_lime(
                meta_model, X_meta, y_pred, base_names, focus_class=focus_class, model_path=meta_model_path
            )

            result = {"status": "success"}

            shap_result = explanations.get("shap", {})
            if shap_result.get("shap_plot"):
                result["shap_plot"] = shap_result["shap_plot"]
            if shap_result.get("shap_importance"):
                result["shap_importance"] = shap_result["shap_importance"]
            if shap_result.get("shap_error"):
                result["shap_error"] = shap_result["shap_error"]

            lime_result = explanations.get("lime", {})
            if lime_result.get("lime_plot"):
                result["lime_plot"] = lime_result["lime_plot"]
            if lime_result.get("lime_example_0"):
                result["lime_example_0"] = lime_result["lime_example_0"]
            if lime_result.get("lime_error"):
                result["lime_error"] = lime_result["lime_error"]

            return result

        # input mode
        parsed_values = []
        for value in input_values:
            if isinstance(value, str):
                if value.upper() == "TRUE":
                    parsed_values.append(1)
                elif value.upper() == "FALSE":
                    parsed_values.append(0)
                else:
                    parsed_values.append(float(value))
            else:
                parsed_values.append(float(value))

        X_raw = np.array([parsed_values], dtype=float)

        base_cols = []
        for m, mp in zip(base_models, base_model_paths):
            X_in = np.array([parsed_values], dtype=np.float32) if str(mp).lower().endswith(".zip") else X_raw
            base_cols.append(_get_meta_feature_for_model(m, X_in, mp, focus_class=focus_class))

        X_meta = np.column_stack(base_cols)
        meta_feature_df = pd.DataFrame(X_meta, columns=base_names)
        save_predictions(meta_feature_df, ".csv", "meta_feature", task_dir)

        pred = meta_model.predict(X_meta)
        pred_list = pred.tolist() if isinstance(pred, np.ndarray) else pred

        out = {"status": "success", "prediction": pred_list}
        try:
            proba = _predict_proba_any(meta_model, X_meta, model_path=meta_model_path)[0]
            num_class = int(len(proba))
            fc = _pick_focus_class(num_class, focus_class)
            out["num_class"] = int(num_class)
            out["focus_class"] = int(fc)
            out["proba"] = proba.tolist()
            out["proba_focus"] = float(proba[fc])
            out["proba_top"] = float(np.max(proba))
        except Exception:
            pass

        return out

    except Exception as e:
        return {"status": "error", "message": f"{e}"}
    finally:
        shutil.rmtree(temp_dir)


# =========================
# Main
# =========================
def main(model_path, mode, task_dir, data_path=None, output_name=None, input_values=None,
         label_column="label", pred_column="prediction", focus_class=1):
    try:
        if model_path.lower().endswith("zip"):
            try:
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    if "stacking_config.json" in zip_ref.namelist():
                        results = predict_stacking(
                            model_path, mode, task_dir,
                            data_path=data_path, output_name=output_name,
                            label_column=label_column, pred_column=pred_column,
                            input_values=input_values, focus_class=focus_class
                        )
                    else:
                        results = predict_single_model(
                            model_path, mode, task_dir,
                            data_path=data_path, output_name=output_name,
                            label_column=label_column, pred_column=pred_column,
                            input_values=input_values, focus_class=focus_class
                        )
            except zipfile.BadZipFile:
                results = {"status": "error", "message": f"Invalid zip file: {model_path}"}
        else:
            results = predict_single_model(
                model_path, mode, task_dir,
                data_path=data_path, output_name=output_name,
                label_column=label_column, pred_column=pred_column,
                input_values=input_values, focus_class=focus_class
            )

        results["task_dir"] = task_dir
        os.makedirs(task_dir, exist_ok=True)
        result_json_path = os.path.join(task_dir, "metrics.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)

        extract_base64_images_and_clean_json(task_dir, "metrics.json")
        print(json.dumps(results))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"{e}"}))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('mode', type=str, choices=["file", "input"])
    parser.add_argument("task_dir", type=str)

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--input_values', type=str, nargs='+')

    # ✅ 保留你原本會傳的 label_column（存在就 drop，不存在也不會錯）
    parser.add_argument('--label_column', type=str, default="label")

    parser.add_argument('--pred_column', type=str, default="prediction")

    # ✅ 多元時要看哪個類別（binary 會自動當 1）
    parser.add_argument('--focus_class', type=int, default=1)

    args = parser.parse_args()
    main(
        args.model_path, args.mode, args.task_dir,
        data_path=args.data_path,
        output_name=args.output_name,
        input_values=args.input_values,
        label_column=args.label_column,
        pred_column=args.pred_column,
        focus_class=args.focus_class
    )
