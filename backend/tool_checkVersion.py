import subprocess
import sys

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
    "matplotlib": "3.10.0"
}

for package, version in packages.items():
    check_and_install(package, version)
