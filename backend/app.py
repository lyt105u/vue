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
    try:
        param = request.json.get('param', None)
        if not param:
            return jsonify({
                "status": "error",
                "message": "缺少必要參數 param"
            }), 400
        fetch_result = subprocess.run(
            ['python', 'fetchData.py', param],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,  # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True  # 將輸出轉換為字符串
        )
        # 取得 fetchData.py 的執行結果
        fetch_output = fetch_result.stdout.strip()
        if fetch_result.returncode != 0:
            # 如果 fetchData.py 執行失敗，回傳錯誤
            return jsonify({
                "status": "error",
                "message": fetch_result.stderr.strip()
            }), 500

        # 回傳檔名列表
        return jsonify({
            "status": "success",
            "files": fetch_output.splitlines()  # 將每行轉為列表
        })
    
    except Exception as e:
        # 如果發生其他錯誤，回傳錯誤訊息
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

    try:
        result = subprocess.check_output(
            'python train.py {} "{}" "{}"'.format(arg1, arg2, arg3),
            shell=True,
            text=True
        )
        # 將 JSON 結果返回給前端
        return jsonify(json.loads(result))
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.output}), 500

if __name__ == '__main__':
    app.run(debug=True)
