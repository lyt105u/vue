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
                print(json.dumps({
                    "status": "error",
                    "message": f"Failed to load TabNet model: {e}",
                }))
                sys.exit(1)

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
