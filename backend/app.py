# python -m venv myenv              # 建立虛擬環境
# myenv\Scripts\activate            # 啟動虛擬環境
# pip install -r requirements.txt   # 安裝依賴

# pip freeze > requirements.txt     # 更新環境版本紀錄



from flask import Flask, request, jsonify, send_file, send_from_directory, after_this_request
from flask_cors import CORS
import subprocess
import json
import os
import mimetypes
import shutil
from datetime import datetime
import base64
import zipfile
import io
from docx import Document
from cleaner import scheduled_clean
import threading

# 建立 Flask 應用，指定靜態資源位置（給 Vue build 後的檔案用）
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, expose_headers=["Content-Disposition"])  # 啟用跨域支持

# 當訪問 "/" 時，回傳 index.html
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# 支援前端 Vue Router：讓 404 fallback 到 index.html
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

# 儲存每個正在執行的 task_id 與對應的子進程物件
process_pool = {}

# 定時清理 model/ 和 upload/，到 cleaner.py 設定時間
t = threading.Thread(target=scheduled_clean, daemon=True)
t.start()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()

    try:
        with open('listWhite.txt', 'r') as f:
            whitelist = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error while loading white list: {e}"
        })

    if username in whitelist:
        return jsonify({
            "status": "success",
            "username": username
        })
    else:
        return jsonify({
            "status": "unauthorized",
            "message": f"{username} is not in the list."
        })

@app.route('/list-files', methods=['POST'])
def list_files():
    data = request.get_json()
    folder_path = data.get('folder_path')
    if not folder_path:
        return jsonify({
            "status": "error",
            "message": "Missing 'folder_path' in request."
        })

    # 取出所有 key 以 ext 開頭的參數（例如 ext1, ext2...）
    # 輸入範例
    # {
    #     "folder_path": "data",
    #     "ext1": "csv",
    #     "ext2": "txt"
    # }
    extensions = []
    for key in sorted(data.keys()):
        if key.startswith("ext") and isinstance(data[key], str) and data[key].strip():
            ext = data[key].strip()
            extensions.append(ext)
    
    try:
        fetch_result = subprocess.run(
            ['python', 'listFiles.py', folder_path] + extensions,
            # capture_output=True,  # 捕獲標準輸出和標準錯誤
            stdout=subprocess.PIPE,     # 只捕獲標準輸出
            stderr=subprocess.DEVNULL,  # 忽略標準錯誤
            text=True                   # 將輸出轉換為字符串
        )
        
        if fetch_result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed or folder not found." + fetch_result.stderr,
            })

        return jsonify(json.loads(fetch_result.stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/get-field-number', methods=['POST'])
def get_field_number():
    model_path = request.json.get('model_path', None)
    if not model_path:
        return jsonify({
            "status": "error",
            "message": "Missing 'param' in request."
        })
    
    try:
        # fetch_result = subprocess.run(
        #     ['python', 'getFieldNumber.py', model_path],
        #     # capture_output=True,  # 捕獲標準輸出和標準錯誤
        #     stdout=subprocess.PIPE,     # 只捕獲標準輸出
        #     stderr=subprocess.DEVNULL,  # 忽略標準錯誤
        #     text=True                   # 將輸出轉換為字符串
        # )
        
        # # debug 用
        # # print("STDOUT:", result.stdout)  # 打印标准输出
        # # print("STDERR:", result.stderr)  # 打印标准错误
        
        # if fetch_result.returncode != 0:
        #     return jsonify({
        #         "status": "error",
        #         "message": fetch_result.stderr,
        #     })

        # return jsonify(json.loads(fetch_result.stdout))

        args = [
            'python', 'getFieldNumber.py',
            model_path
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)
        if p.returncode != 0:
            return jsonify({
                "status": "error",
                "message": stderr
            })
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-xgb', methods=['POST'])
def run_train_xgb():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')

    try:
        # result = subprocess.run(
        #     ['python', 'train_xgb.py', file_path, label_column, split_strategy, split_value, model_name, n_estimators, learning_rate, max_depth],
        #     # capture_output=True,        # 捕獲標準輸出和標準錯誤
        #     stdout=subprocess.PIPE,     # 只捕獲標準輸出
        #     stderr=subprocess.DEVNULL,  # 忽略標準錯誤
        #     text=True                   # 將輸出轉換為字符串
        # )

        # # print("STDOUT:", result.stdout)  # 印出標準輸出
        # # print("STDERR:", result.stderr)  # 印出標準錯誤

        # if result.returncode != 0:
        #     return jsonify({
        #         "status": "error",
        #         "message": result.stderr,
        #     })
        
        # return jsonify(json.loads(result.stdout))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "xgb",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        args = [
            'python', 'train_xgb.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, learning_rate, max_depth,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-lgbm', methods=['POST'])
def run_train_lgbm():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    max_depth = data.get('max_depth')
    num_leaves = data.get('num_leaves')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "lgbm",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_lgbm.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, learning_rate, max_depth, num_leaves,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-rf', methods=['POST'])
def run_train_rf():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    max_depth = data.get('max_depth')
    random_state = data.get('random_state')
    n_jobs = data.get('n_jobs')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "rf",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_rf.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, max_depth, random_state, n_jobs,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-lr', methods=['POST'])
def run_train_lr():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    penalty = data.get('penalty')
    C = data.get('C')
    max_iter = data.get('max_iter')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "lr",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_lr.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            penalty, C, max_iter,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-tabnet', methods=['POST'])
def run_train_tabnet():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    batch_size = data.get('batch_size')
    max_epochs = data.get('max_epochs')
    patience = data.get('patience')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "tabnet",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_tabnet.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            batch_size, max_epochs, patience,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-mlp', methods=['POST'])
def run_train_mlp():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    hidden_layer_1 = data.get('hidden_layer_1')
    hidden_layer_2 = data.get('hidden_layer_2')
    hidden_layer_3 = data.get('hidden_layer_3')
    activation = data.get('activation')
    learning_rate_init = data.get('learning_rate_init')
    max_iter = data.get('max_iter')
    n_iter_no_change = data.get('n_iter_no_change')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "mlp",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_mlp.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            hidden_layer_1, hidden_layer_2, hidden_layer_3,
            activation, learning_rate_init, max_iter, n_iter_no_change,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-train-catboost', methods=['POST'])
def run_train_catboost():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    iterations = data.get('iterations')
    learning_rate = data.get('learning_rate')
    depth = data.get('depth')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "catboost",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_catboost.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            iterations, learning_rate, depth,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-adaboost', methods=['POST'])
def run_train_adaboost():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    n_estimators = data.get('n_estimators')
    learning_rate = data.get('learning_rate')
    depth = data.get('depth')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "adaboost",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_adaboost.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            n_estimators, learning_rate, depth,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/run-train-svm', methods=['POST'])
def run_train_svm():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    split_strategy = data.get('split_strategy') 
    split_value = data.get('split_value')
    model_name = data.get('model_name')
    username = data.get('username')
    C = data.get('C')
    kernel = data.get('kernel')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "svm",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_svm.py',
            file_path, label_column, split_strategy,
            split_value, model_name,
            C, kernel,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
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
    username = data.get('username')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"{timestamp}"
    task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/

    if not model_path or not mode:
        return jsonify({"status": "error", "message": "model_path and mode are necessary args"})

    command = ['python', 'predict.py', model_path, mode, task_dir]
    if mode == 'file':
        if not data_path or not output_name:
            return jsonify({"status": "error", "message": "data_path 和 output_name 是 file 模式下的必要參數"})
        command.extend(['--data_path', data_path, '--output_name', output_name, '--label_column', label_column])
    elif mode == 'input':
        if not input_values or not isinstance(input_values, list):
            return jsonify({"status": "error", "message": "input_values 是 input 模式下的必要參數，且必須為陣列格式"})
        command.extend(['--input_values'] + list(map(str, input_values)))

    try:
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "predict",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        # 使用 Popen 開啟子進程
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        # 等待子進程結束並接收輸出
        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/run-evaluate', methods=['POST'])
def run_evaluate():
    data = request.get_json()
    model_path = data.get('model_path')  # 模型路徑
    data_path = data.get('data_path')   # 輸入檔案路徑
    output_name = data.get('output_name')  # 輸出檔案名稱
    label_column = data.get('label_column')
    pred_column = data.get('pred_column')
    username = data.get('username')
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "evaluate",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'evaluate.py',
            model_path, data_path, output_name, label_column, pred_column,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/delete-tabular-rows', methods=['POST'])
def delete_tabular_rows():
    data = request.get_json()
    file_path = data.get('file_path')
    rows = data.get('rows')

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
    
@app.route('/preview-tabular', methods=['POST'])
def preview():
    data = request.get_json()
    file_path = data.get('file_path')
    try:
        fetch_result = subprocess.run(
            ['python', 'checkPreviewTab.py', file_path],
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
    
@app.route('/preprocess', methods=['POST'])
def handle_missing():
    data = request.get_json()
    file_path = data.get('file_path')
    rules = data.get("rules")
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": "File not found."
        })
    try:
        fetch_result = subprocess.run(
            ['python', 'preprocess.py', file_path, json.dumps(rules)],
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

@app.route('/upload-local-file', methods=['POST'])
def upload_local_file():
    if 'file' not in request.files or 'folder' not in request.form:
        return jsonify({
            "status": "error",
            "message": "Missing file or folder in request."
        })
    
    file = request.files['file']
    folder = request.form['folder']
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected."
        })
    
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, file.filename)
    file.save(save_path)
    return jsonify({
        "status": "success"
    })

@app.route('/upload-samba-file', methods=['POST'])
def upload_samba_file():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    remote_path = data.get('remote_path')
    folder = data.get('folder')

    try:
        fetch_result = subprocess.run(
            ['python', 'uploadSambaFile.py', username, password, remote_path, folder],
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

@app.route('/copy-local-file', methods=['POST'])
def copy_local_file():
    data = request.get_json()
    source_path = data.get('source_path')
    target_folder = data.get('target_folder')

    if not source_path or not target_folder:
        return jsonify({
            "status": "error",
            "message": "Missing source_path or target_folder."
        })

    if not os.path.isfile(source_path):
        return jsonify({
            "status": "error",
            "message": f"Source file does not exist: {source_path}"
        })

    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.basename(source_path)
    target_path = os.path.join(target_folder, filename)

    try:
        shutil.copy2(source_path, target_path)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

    return jsonify({
        "status": "success",
    })

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        task_dir = data.get("task_dir")

        if not task_dir or not os.path.exists(task_dir):
            return jsonify({"status": "error", "message": "task_dir not found"}), 404

        # 在記憶體中建立 ZIP 檔
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(task_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=task_dir)
                    zipf.write(full_path, arcname)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=os.path.basename(task_dir) + ".zip"
        )

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
    
@app.route('/download-stacking-models', methods=['POST'])
def download_stacking_models():
    try:
        data = request.get_json()
        task_dir = data.get("task_dir")

        if not task_dir or not os.path.exists(task_dir):
            return jsonify({"status": "error", "message": "task_dir not found"})

        config_path = os.path.join(task_dir, "stacking_config.json")
        if not os.path.exists(config_path):
            return jsonify({"status": "error", "message": "stacking_config.json not found"})

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        files_to_include = [config_path]

        # 處理 base model 檔名
        for base_name in config.get("base_models", []):
            if base_name == "xgb":
                filename = f"base_{base_name}.json"
            elif base_name == "tabnet":
                filename = f"base_{base_name}.zip"
            else:
                filename = f"base_{base_name}.pkl"
            full_path = os.path.join(task_dir, filename)
            if os.path.exists(full_path):
                files_to_include.append(full_path)
            else:
                return jsonify({"status": "error", "message": f"{filename} not found"})

        # 處理 meta model 檔名
        meta_name = config.get("meta_model")
        if meta_name:
            if meta_name == "xgb":
                filename = f"meta_{meta_name}.json"
            elif meta_name == "tabnet":
                filename = f"meta_{meta_name}.zip"
            else:
                filename = f"meta_{meta_name}.pkl"
            full_path = os.path.join(task_dir, filename)
            if os.path.exists(full_path):
                files_to_include.append(full_path)
            else:
                return jsonify({"status": "error", "message": f"{filename} not found"})

        # 打包成 ZIP 檔
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for full_path in files_to_include:
                arcname = os.path.basename(full_path)
                zipf.write(full_path, arcname)

        zip_buffer.seek(0)
        zip_filename = os.path.basename(task_dir) + "_stacking_models.zip"

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/check-label-uniqueness', methods=['POST'])
def check_label_uniqueness():
    data = request.get_json()
    file_path = data.get('file_path')
    label_column = data.get('label_column')
    try:
        fetch_result = subprocess.run(
            ['python', 'checkLabelUniqueness.py', file_path, label_column],
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
    
@app.route('/delete-files', methods=['POST'])
def delete_files():
    data = request.get_json()
    folder_path = data.get('folder_path')  # e.g., "upload/username"
    files = data.get('files', [])          # e.g., ["a.csv", "b.xlsx"]
    if not folder_path or not files:
        return jsonify({
            'status': 'error',
            'message': 'Missing folder_path or files list.'
        })

    deleted = []
    failed = []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted.append(filename)
            else:
                failed.append((filename, "File not found"))
        except Exception as e:
            failed.append((filename, str(e)))

    if failed:
        return jsonify({
            'status': 'error',
            'message': f"Some files could not be deleted: {failed}",
            'deleted': deleted
        })

    return jsonify({
        'status': 'success',
        'message': f"Deleted {len(deleted)} files.",
        'deleted': deleted
    })

# 抓出使用者資料夾內，每個 task_dir 內 status.json 的內容
@app.route('/list-status', methods=['POST'])
def list_status():
    data = request.get_json()
    user_dir = data.get('user_dir')  # e.g. model/alice

    if not user_dir:
        return jsonify({
            "status": "error",
            "message": "Missing 'user_dir' in request."
        })

    try:
        result = subprocess.run(
            ['python', 'listStatus.py', user_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Script execution failed."
            })

        return jsonify(json.loads(result.stdout))

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# 抓出該 task_dir 中 status.json 的內容
@app.route('/get-status', methods=['POST'])
def get_status():
    data = request.get_json()
    folder_path = data.get('folder_path')

    if not folder_path:
        return jsonify({
            "status": "error",
            "message": "Missing 'folder_path' in request."
        })

    status_path = os.path.join(folder_path, "status.json")
    if not os.path.exists(status_path):
        return jsonify({
            "status": "error",
            "message": f"'status.json' not found in folder: {folder_path}"
        })

    try:
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        return jsonify({
            "status": "success",
            "data": status_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to read status.json: {str(e)}"
        })

@app.route('/delete-folders', methods=['POST'])
def delete_folders():
    data = request.get_json()
    folder_path = data.get('folder_path')   # e.g., "model/alice"
    folders = data.get('folders', [])       # e.g., ["a", "b", "c"]

    if not folder_path or not folders:
        return jsonify({
            'status': 'error',
            'message': 'Missing folder_path or folders list.'
        }), 400

    deleted = []
    failed = []

    for subfolder in folders:
        full_path = os.path.join(folder_path, subfolder)
        try:
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
                deleted.append(subfolder)
            else:
                failed.append((subfolder, "Folder not found"))
        except Exception as e:
            failed.append((subfolder, str(e)))

    if failed:
        return jsonify({
            'status': 'error',
            'message': f"Some folders could not be deleted: {failed}",
            'deleted': deleted
        })

    return jsonify({
        'status': 'success',
        'message': f"Deleted {len(deleted)} folders.",
        'deleted': deleted
    })

@app.route('/terminate-task', methods=['POST'])
def terminate_task():
    data = request.get_json()
    task_id = data.get("task_id")
    username = data.get("username")  # 驗證使用者身分（如有登入系統）

    task_info = process_pool.get(task_id)

    if not task_info:
        return jsonify({
            "status": "notRunning",
            "message": f"Task ID '{task_id}' is not running."
        })

    if username and task_info.get("username") != username:
        return jsonify({
            "status": "error",
            "message": "Permission denied: task belongs to another user."
        })

    process = task_info["process"]
    task_dir = task_info["task_dir"]
    status_path = os.path.join(task_dir, "status.json")

    try:
        # 終止子進程
        process.terminate()
        process_pool.pop(task_id, None)

        # 更新 status.json
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
        else:
            status = {"task_id": task_id}

        status["status"] = "terminated"
        status["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        return jsonify({
            "status": "success",
            "message": f"Task {task_id} terminated."
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to terminate task: {str(e)}"
        })
    
@app.route('/run-train-stacking', methods=['POST'])
def run_train_stacking():
    data = request.get_json()
    base_models = data.get('base_models', [])
    data_name = data.get('data_name')
    label_column = data.get('label_column')
    meta_model = data.get('meta_model')
    username = data.get('username')

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{timestamp}"
        task_dir = os.path.join("model", username, timestamp)   # model/<username>/<timestamp>/
        os.makedirs(task_dir, exist_ok=True)
        # status.json
        status = {
            "task_id": task_id,
            "username": username,
            "status": "running",
            "params": data,
            "start_time": timestamp,
            "api": "stacking",
        }
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)

        args = [
            'python', 'train_stacking.py',
            data_name, json.dumps(base_models), label_column, meta_model,
            task_dir
        ]
        # 執行 Python 訓練腳本（非阻塞，可終止）
        p = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        process_pool[task_id] = {
            "process": p,
            "username": username,
            "task_dir": task_dir
        }

        stdout, stderr = p.communicate()
        # --- DEBUG 輸出（如需印出，請取消註解） ---
        # print("STDOUT:", stdout, flush=True)
        # print("STDERR:", stderr, flush=True)

        # 如果 task_id 不在 process_pool，代表已被終止
        if task_id not in process_pool:
            # 不寫 status，避免覆蓋已寫入的 terminated 狀態
            return jsonify({
                "status": "terminated",
                "message": "Task was terminated by user.",
                "task_id": task_id
            })
        result = json.loads(stdout)
        if result.get("status") == "error" or p.returncode != 0:
            # 更新 status.json 為 error
            status['status'] = 'error'
            status['msg'] = result.get("message") or stderr or "Unknown error."
            with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
            # 任務完成後移除該 process 記錄
            process_pool.pop(task_id, None)
            return jsonify({
                "status": "error",
                "message": result.get("message") or stderr or "Unknown error."
            })
        # 更新 status.json 為 success
        status['status'] = 'success'
        status['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        # stacking_config.json
        stacking_config = {
            "base_models": base_models,
            "meta_model": meta_model,
        }
        with open(os.path.join(task_dir, "stacking_config.json"), "w", encoding="utf-8") as f:
            json.dump(stacking_config, f, indent=4, ensure_ascii=False)
        return jsonify(json.loads(stdout))
    
    except Exception as e:
        # 更新 status.json 為 error
        status['status'] = 'error'
        status['msg'] = e or "Unknown error."
        with open(os.path.join(task_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        # 任務完成後移除該 process 記錄
        process_pool.pop(task_id, None)
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
