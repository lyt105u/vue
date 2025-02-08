# Tool
# 用來 xlsx 和 csv 的 label 欄位 （不給看答案），用於 predict 頁面
# 先將有 label 的 xlsx 和 csv 移到 predict 資料夾，再下指令：python tool_removeLabelColumn.py

import os
import pandas as pd

# 指定目錄路徑
folder_path = "data/predict"

# 確保目錄存在
if not os.path.exists(folder_path):
    print(f"目錄 {folder_path} 不存在！")
    exit()

# 處理每個檔案
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # 確認檔案類型（CSV 或 Excel）
    if file_name.endswith('.csv'):
        # 讀取 CSV 檔案
        df = pd.read_csv(file_path)
    elif file_name.endswith(('.xls', '.xlsx')):
        # 讀取 Excel 檔案
        df = pd.read_excel(file_path)
    else:
        continue  # 跳過非 CSV/Excel 檔案

    # 如果 `label` 欄位存在則刪除
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
        # 另存新檔（覆蓋原檔案）
        if file_name.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)
        print(f"已處理並儲存：{file_name}")
    else:
        print(f"未發現 label 欄位：{file_name}")

print("處理完成！")
