# python -m venv myenv              # 建立虛擬環境
# myenv\Scripts\activate            # 啟動虛擬環境
# pip install -r requirements.txt   # 安裝依賴

# pip freeze > requirements.txt     # 更新環境版本紀錄



from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import json
import os
import mimetypes

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
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

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
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-xgb', methods=['POST'])
def run_train_xgb():
    data = request.get_json()
    file_name = data.get('file_name')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')

    try:
        result = subprocess.run(
            ['python', 'train_xgb.py', file_name, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 印出標準輸出
        # print("STDERR:", result.stderr)  # 印出標準錯誤

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr,
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-lgbm', methods=['POST'])
def run_train_lgbm():
    data = request.get_json()
    file_name = data.get('file_name')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')
    num_leaves = data.get('num_leaves')

    try:
        result = subprocess.run(
            ['python', 'train_lgbm.py', file_name, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth, num_leaves],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 印出標準輸出
        # print("STDERR:", result.stderr)  # 印出標準錯誤

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-rf', methods=['POST'])
def run_train_rf():
    data = request.get_json()
    file_name = data.get('file_name')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    n_estimators = data.get('n_estimators')
    max_depth = data.get('max_depth')
    random_state = data.get('random_state')
    n_jobs = data.get('n_jobs')

    try:
        result = subprocess.run(
            ['python', 'train_rf.py', file_name, label_column, split_strategy, split_value, model_name, n_estimators, max_depth, random_state, n_jobs],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 印出標準輸出
        # print("STDERR:", result.stderr)  # 印出標準錯誤

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-lr', methods=['POST'])
def run_train_lr():
    data = request.get_json()
    file_name = data.get('file_name')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    penalty = data.get('penalty')
    C = data.get('C')
    solver = data.get('solver')
    max_iter = data.get('max_iter')

    try:
        result = subprocess.run(
            ['python', 'train_lr.py', file_name, label_column, split_strategy, split_value, model_name, penalty, C, solver, max_iter],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 印出標準輸出
        # print("STDERR:", result.stderr)  # 印出標準錯誤

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-tabnet', methods=['POST'])
def run_train_tabnet():
    data = request.get_json()
    file_name = data.get('file_name')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    batch_size = data.get('batch_size')
    max_epochs = data.get('max_epochs')
    patience = data.get('patience')

    try:
        result = subprocess.run(
            ['python', 'train_tabnet.py', file_name, label_column, split_strategy, split_value, model_name, batch_size, max_epochs, patience],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 印出標準輸出
        # print("STDERR:", result.stderr)  # 印出標準錯誤

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-mlp', methods=['POST'])
def run_train_mlp():
    data = request.get_json()
    file_name = data.get('file_name')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    hidden_layer_1 = data.get('hidden_layer_1')
    hidden_layer_2 = data.get('hidden_layer_2')
    hidden_layer_3 = data.get('hidden_layer_3')
    activation = data.get('activation')
    learning_rate_init = data.get('learning_rate_init')
    max_iter = data.get('max_iter')

    try:
        result = subprocess.run(
            ['python', 'train_mlp.py', file_name, label_column, split_strategy, split_value, model_name, hidden_layer_1, hidden_layer_2, hidden_layer_3, activation, learning_rate_init, max_iter],
            # capture_output=True,        # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )

        # print("STDOUT:", result.stdout)  # 印出標準輸出
        # print("STDERR:", result.stderr)  # 印出標準錯誤

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-predict', methods=['POST'])
def run_predict():
    data = request.get_json()
    model_path = data.get('model_path')  # 模型路徑
    mode = data.get('mode')             # 模式：file 或 input
    data_path = data.get('data_path')   # 輸入檔案路徑（file 模式）
    output_name = data.get('output_name')  # 輸出檔案名稱（file 模式）
    input_values = data.get('input_values')  # 特徵值陣列（input 模式）
    label_column = data.get('label_column')

    if not model_path or not mode:
        return jsonify({"status": "error", "message": "model_path and mode are necessary args"}), 400

    command = ['python', 'predict.py', model_path, mode]
    if mode == 'file':
        if not data_path or not output_name:
            return jsonify({"status": "error", "message": "data_path 和 output_name 是 file 模式下的必要參數"}), 400
        command.extend(['--data_path', data_path, '--output_name', output_name, '--label_column', label_column])
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
                "message": result.stderr
            })
        
        return jsonify(json.loads(result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/upload-Tabular', methods=['POST'])
def upload_and_check():
    UPLOAD_FOLDER = 'data/upload'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "Missing file in request."
        })
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected."
        })
    
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    
    try:
        fetch_result = subprocess.run(
            ['python', 'checkPreviewTab.py', file.filename],
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
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/delete-Tabular-Rows', methods=['POST'])
def delete_tabular_rows():
    data = request.get_json()
    filename = data.get('filename')
    rows = data.get('rows')

    UPLOAD_FOLDER = 'data/upload'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        fetch_result = subprocess.run(
            ['python', 'deleteTabRows.py', file_path, json.dumps(rows)],
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        # debug 用
        # print("STDOUT:", fetch_result.stdout)  # 打印标准输出
        # print("STDERR:", fetch_result.stderr)  # 打印标准错误
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/preview-Tabular', methods=['POST'])
def preview():
    data = request.get_json()
    filename = data.get('filename')
    try:
        fetch_result = subprocess.run(
            ['python', 'checkPreviewTab.py', filename],
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
                "message": fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/upload-Model', methods=['POST'])
def upload_model():
    UPLOAD_FOLDER = 'model'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "Missing file in request."
        })
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected."
        })
    
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    return jsonify({
        "status": "success",
    })
    
@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    download_path = data.get('download_path')

    if not download_path or not os.path.exists(download_path):
        return jsonify({
            "status": "error",
            "message": "Download path not found."
        }), 404

    # 自動判斷 mimetype（根據副檔名）
    mime_type, _ = mimetypes.guess_type(download_path)
    if not mime_type:
        mime_type = 'application/octet-stream'  # 二進位 fallback

    filename = os.path.basename(download_path)
    return send_file(
        download_path,
        mimetype=mime_type,
        as_attachment=True,
        download_name=filename
    )

@app.route('/download-Smb', methods=['POST'])
def downloaSmb():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    remote_path = data.get('remote_path')
    local_path = 'data/upload'

    try:
        fetch_result = subprocess.run(
            ['python', 'downloadSmb.py', username, password, remote_path, local_path],
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
                "message": fetch_result,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
