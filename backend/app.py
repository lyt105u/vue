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

@app.route('/run-predict', methods=['POST'])
def run_predict():
    data = request.get_json()
    arg1 = data.get('arg1')
    arg2 = data.get('arg2')
    arg3 = data.get('arg3')

    try:
        # result = subprocess.check_output(['python', 'predict.py', arg1, arg2], text=True)
        result = subprocess.check_output(
            'python predict.py {} "{}" "{}"'.format(arg1, arg2, arg3),
            shell=True,
            text=True
        )
        # 將 JSON 結果返回給前端
        return jsonify(json.loads(result))
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.output}), 500
    
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

if __name__ == '__main__':
    app.run(debug=True)
