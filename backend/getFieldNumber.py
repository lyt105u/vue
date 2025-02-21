import sys
import os
import json
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib

def get_field_number(model_name):
    try:
        model_path = os.path.join("model", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_path}' doesn't exist!")

        # 確定檔案類型
        file_extension = os.path.splitext(model_path)[-1].lower()

        # 讀取 XGBoost 模型（JSON 格式）
        if file_extension == ".json":
            try:
                xgb_model = XGBClassifier()
                xgb_model.load_model(model_path)
                return xgb_model.n_features_in_
            except Exception as e:
                raise ValueError(f"Failed to load XGBoost model: {e}")

        # 讀取 TabNet 模型（需 `.zip` 格式）
        elif file_extension == ".zip":
            try:
                tabnet_model = TabNetClassifier()
                tabnet_model.load_model(model_path)
                return tabnet_model.network.input_dim
            except Exception as e:
                raise ValueError(f"Failed to load TabNet model: {e}")

        # 讀取其他 Scikit-Learn 及 LightGBM 模型（PKL 格式）
        elif file_extension == ".pkl":
            try:
                model = joblib.load(model_path)

                # Scikit-learn, LightGBM
                if hasattr(model, "n_features_in_"):
                    return model.n_features_in_
                else:
                    raise ValueError("The loaded model does not have 'n_features_in_' attribute.")
            except Exception as e:
                raise ValueError(f"Failed to load Pickle/Joblib model: {e}")

        else:
            raise ValueError(f"Unsupported format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error in get_field_number: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python getFieldNumber.py <model_name>")

    model_name = sys.argv[1]
    field_count = get_field_number(model_name)

    print(json.dumps({
        "status": "success",
        "field_count": field_count
    }))
