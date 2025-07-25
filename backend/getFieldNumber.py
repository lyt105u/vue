# 取得模型所需要的欄位數
# usage: python getFieldNumber.py <model_path>
# ex:
#   python checkPreviewTab.py upload/tabular.csv

import sys
import os
import json
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import zipfile
import tempfile

def get_field_number(model_path):
    try:
        if not os.path.exists(model_path):
            print(json.dumps({
                "status": "error",
                "message": f"Model '{model_path}' doesn't exist!",
            }))
            sys.exit(1)

        # 確定檔案類型
        file_extension = os.path.splitext(model_path)[-1].lower()

        # 讀取 XGBoost 模型（JSON 格式）
        if file_extension == ".json":
            try:
                xgb_model = XGBClassifier()
                xgb_model.load_model(model_path)
                return xgb_model.n_features_in_
            except Exception as e:
                print(json.dumps({
                    "status": "error",
                    "message": f"Failed to load XGBoost model: {e}",
                }))
                sys.exit(1)

        # 讀取 TabNet 模型（需 `.zip` 格式）
        elif file_extension == ".zip":
            try:
                tabnet_model = TabNetClassifier()
                tabnet_model.load_model(model_path)
                return tabnet_model.network.input_dim
            except Exception as e:
                # 如果不是 TabNet，嘗試當作 stacking ZIP 模型
                return get_field_number_from_stacking_zip(model_path)

        # 讀取其他 Scikit-Learn 及 LightGBM 模型（PKL 格式）
        elif file_extension == ".pkl":
            try:
                model = joblib.load(model_path)

                # Scikit-learn, LightGBM
                if hasattr(model, "n_features_in_"):
                    return model.n_features_in_
                else:
                    print(json.dumps({
                        "status": "error",
                        "message": "The loaded model does not have 'n_features_in_' attribute.",
                    }))
                    sys.exit(1)
            except Exception as e:
                print(json.dumps({
                    "status": "error",
                    "message": f"Failed to load Pickle/Joblib model: {e}",
                }))
                sys.exit(1)

        else:
            print(json.dumps({
                "status": "error",
                "message": f"Unsupported format: {file_extension}",
            }))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Error in get_field_number: {e}",
        }))
        sys.exit(1)

def get_field_number_from_stacking_zip(zip_path):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 解壓縮 zip 到暫存目錄
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # 讀取 stacking_config.json
            config_path = os.path.join(tmpdir, "stacking_config.json")
            if not os.path.exists(config_path):
                print(json.dumps({
                    "status": "error",
                    "message": "Missing stacking_config.json in ZIP archive.",
                }))
                sys.exit(1)

            with open(config_path, "r") as f:
                config = json.load(f)

            base_model_paths = config.get("base_models", [])
            if not base_model_paths:
                print(json.dumps({
                    "status": "error",
                    "message": "No base models listed in stacking_config.json.",
                }))
                sys.exit(1)

            # 取第一個 base model 來取得欄位數
            raw_path = base_model_paths[0]
            candidate_paths = [
                os.path.join(tmpdir, "base_" + raw_path + ".pkl"),
                os.path.join(tmpdir, "base_" + raw_path + ".json"),
                os.path.join(tmpdir, "base_" + raw_path + ".zip"),
            ]

            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    return get_field_number(candidate)
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Failed to process stacking ZIP model: {e}",
        }))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python getFieldNumber.py <model_path>.",
        }))
        sys.exit(1)

    model_path = sys.argv[1]
    field_count = get_field_number(model_path)

    print(json.dumps({
        "status": "success",
        "field_count": field_count
    }))
