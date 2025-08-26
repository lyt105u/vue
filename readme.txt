# MLAS - 機器學習分析系統  
  
## 系統概述  
MLAS 是一個全面的網頁式平台，提供端到端的機器學習功能，專門用於表格式資料分析。系統讓使用者能夠上傳資料集、訓練多種類型的機器學習模型、進行預測，並透過可解釋人工智慧功能評估模型效能。  
  
## 系統架構  
- 前端：Vue.js 單頁應用程式，採用元件化架構  
- 後端：Flask 網頁伺服器，提供 API 端點  
- 機器學習處理：透過子程序管理執行 Python 腳本  
- 儲存：使用者專屬目錄組織，用於資料和模型管理  
  
## 安裝與部署  
參考 deploy_docker.txt 或 deploy.py，會在本地端 mount 起來 bacnked/upload/、bacnked/model/、bacnked/listWhite.txt
  
## 檔案結構  
- backend/upload/{username}/：使用者資料檔案  
- backend/model/{username}/{timestamp}/：訓練好的模型和結果  
- status.json：任務執行追蹤  
  
## I18n  
目前支援英文和中文，涵蓋所有介面元素、表單驗證和說明文字。未來若要新增只需增加該語言 label json，並新增語言選項    

## 登入
白名單為 bacnked/listWhite.txt，要新增用戶可直接新增

## XAI  
每個 model 的 XAI 都寫在各自的腳本哩，因應每個 model 的資料格式不同   