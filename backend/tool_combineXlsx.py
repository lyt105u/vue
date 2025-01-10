# Tool
# 用來合併"川崎症xslx"陽性陰性資料，並分別儲存成xlsx和csv
# 順便做 one-hot encoding 和年齡篩選
# 注意檔名和路徑

import pandas as pd
from sklearn.utils import shuffle
import os

# 長庚
# file1 = "川崎症病人(訓練用 第一階段).xlsx"  # 陽性
# file0 = "發燒病人(訓練用 第一階段).xlsx"    # 陰性
# output_file = "data\長庚"

# 高醫
file1 = "KD_validation.xlsx"  # 陽性
file0 = "FC_validation.xlsx"  # 陰性
output_file = "data\高醫"

pd.options.display.max_columns = None

df1 = pd.read_excel(file1)
df1 = df1[df1['年齡(日)'] <= 1826]
df1['輸入日期(月)'] = df1['輸入日期'].dt.month
df1_month = pd.get_dummies(df1['輸入日期(月)'], prefix='Month')
df1 = pd.concat([df1, df1_month], axis=1).drop(columns=['輸入日期(月)', '輸入日期'])
df1 = shuffle(df1, random_state=30)
df1["label"] = 1

df0 = pd.read_excel(file0)
df0 = df0[df0['年齡(日)'] <= 1826]
df0['輸入日期(月)'] = df0['輸入日期'].dt.month
df0_month = pd.get_dummies(df0['輸入日期(月)'], prefix='Month')
df0 = pd.concat([df0, df0_month], axis=1).drop(columns=['輸入日期(月)', '輸入日期'])
df0 = shuffle(df0, random_state=30)
df0["label"] = 0

df_combined = pd.concat([df1, df0], ignore_index=True)
df_combined.to_excel(f"{output_file}.xlsx", index=False)
df_combined.to_csv(f"{output_file}.csv", index=False, encoding='utf-8')

xlsx_path = os.path.abspath(f"{output_file}.xlsx")
csv_path = os.path.abspath(f"{output_file}.csv")

print("Merge completed. Files have been saved at:")
print(f"- Excel:\t{xlsx_path}")
print(f"- CSV:\t{csv_path}")
