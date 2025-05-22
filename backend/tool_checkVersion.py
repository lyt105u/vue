import subprocess
import sys
import os

def check_and_install(package, version):
    try:
        # Import the package and check the version
        pkg = __import__(package)
        installed_version = pkg.__version__
        if installed_version != version:
            print(f"{package} version mismatch: {installed_version} != {version}. Installing specified version.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        else:
            print(f"{package} {version} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing version {version}.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

# Package list with versions
packages = {
    "pandas": "2.2.3",
    "numpy": "1.26.4",
    "xgboost": "2.1.3",
    "scikit-learn": "1.5.2",
    "lightgbm": "4.5.0",
    "joblib": "1.4.2",
    "matplotlib": "3.10.0",
    "openpyxl": "3.1.5",
}

for package, version in packages.items():
    check_and_install(package, version)

# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "scikit-learn==1.5.2"],
#     stdout=subprocess.DEVNULL
# )
# subprocess.check_call(
#     [sys.executable, "-m", "pip", "install", "--upgrade", "xgboost"],
#     stdout=subprocess.DEVNULL
# )

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 刪除檔案或符號連結
            elif os.path.isdir(file_path):
                # 若資料夾中還有子資料夾則遞迴刪除
                import shutil
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# 指定要清空的資料夾
clear_folder('upload')
clear_folder('model')
