import sys
import os
import json

def get_field_number(model_name):
    try:
        field_count_file = "model/field_counts.json"

        if not os.path.exists(field_count_file):
            raise FileNotFoundError(f"Field count file not found: {field_count_file}")

        with open(field_count_file, "r") as f:
            field_counts = json.load(f)

        if model_name in field_counts:
            return field_counts[model_name]

        raise ValueError(f"Model {model_name} not found in field count file.")

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "message": "Usage: python getFieldNumber.py <model_name>"
        }))
        sys.exit(1)

    model_name = sys.argv[1]
    field_count = get_field_number(model_name)

    print(json.dumps({
        "status": "success",
        "field_count": field_count
    }))
