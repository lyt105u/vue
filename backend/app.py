# python -m venv myenv              # 建立虛擬環境
# myenv\Scripts\activate            # 啟動虛擬環境
# pip install -r requirements.txt   # 安裝依賴

# pip freeze > requirements.txt     # 更新環境版本紀錄



from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json

app = Flask(__name__)
CORS(app)  # 啟用跨域支持

@app.route('/fetch-data', methods=['POST'])
def run_fetch_data():
    param = request.json.get('param', None)
    if not param:
        return jsonify({
            "status": "error",
            "message": "Missing 'param' in request."
        }), 400
    
    try:
        fetch_result = subprocess.run(
            ['python', 'fetchData.py', param],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed.",
                "output": fetch_result.stderr
            }), 500

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/get-fieldNumber', methods=['POST'])
def run_get_fieldNumber():
    param = request.json.get('param', None)
    if not param:
        return jsonify({
            "status": "error",
            "message": "Missing 'param' in request."
        }), 400
    
    try:
        fetch_result = subprocess.run(
            ['python', 'getFieldNumber.py', param],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        # debug 用
        # print("STDOUT:", result.stdout)  # 打印标准输出
        # print("STDERR:", result.stderr)  # 打印标准错误
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed.",
                "output": fetch_result.stderr
            }), 500

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/run-train', methods=['POST'])
def run_train():
    data = request.get_json()
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    arg3 = data.get('arg3')
    arg4 = data.get('arg4')
    arg5 = data.get('arg5')
    arg6 = data.get('arg6')

    try:
        result = subprocess.run(
            ['python', 'train.py', arg1, arg2, arg3, arg4, arg5, arg6],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 打印标准输出
        # print("STDERR:", result.stderr)  # 打印标准错误
        
        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Error occurred while executing train.py.",
                "output": result.stderr
            }), 500
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/run-predict', methods=['POST'])
def run_predict():
    data = request.get_json()
    model_path = data.get('model_path')  # 模型路徑
    mode = data.get('mode')             # 模式：file 或 input
    data_path = data.get('data_path')   # 輸入檔案路徑（file 模式）
    output_name = data.get('output_name')  # 輸出檔案名稱（file 模式）
    input_values = data.get('input_values')  # 特徵值陣列（input 模式）

    if not model_path or not mode:
        return jsonify({"status": "error", "message": "model_path and mode are necessary args"}), 400

    command = ['python', 'predict.py', model_path, mode]
    if mode == 'file':
        if not data_path or not output_name:
            return jsonify({"status": "error", "message": "data_path 和 output_name 是 file 模式下的必要參數"}), 400
        command.extend(['--data_path', data_path, '--output_name', output_name])
    elif mode == 'input':
        if not input_values or not isinstance(input_values, list):
            return jsonify({"status": "error", "message": "input_values 是 input 模式下的必要參數，且必須為陣列格式"}), 400
        command.extend(['--input_values'] + list(map(str, input_values)))

    try:
        # 呼叫子進程執行腳本
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # print("STDOUT:", result.stdout)  # 打印标准输出
        # print("STDERR:", result.stderr)  # 打印标准错误

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Error occurred while executing predict.py.",
                "stderr": result.stderr
            }), 500
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
