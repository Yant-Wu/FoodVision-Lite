# FoodVision Lite

## 簡介
FoodVision Lite 是一個基於深度學習的食物辨識應用程式，使用 PyTorch 和 Streamlit 實現。此專案可以辨識食物圖片並提供其熱量資訊。

## 功能
- 上傳食物圖片進行辨識
- 提供食物的信心分數
- 顯示食物的預估熱量

## 使用方式

### 1. 訓練模型
執行 `train.py` 來訓練模型：
```bash
python train.py
```
訓練完成後，模型會儲存於 `model/food_model.pth`。

### 2. 啟動應用程式
執行 `app.py` 啟動 Streamlit 應用程式：
```bash
streamlit run app.py
```

### 3. 上傳圖片
在應用程式中上傳食物圖片，系統會顯示辨識結果、信心分數以及熱量資訊。

## 專案結構
- `train.py`：模型訓練程式
- `app.py`：Streamlit 應用程式
- `food_utils.py`：包含模型載入、圖片預測及熱量查詢的工具函式
- `calories_lookup.json`：食物熱量資訊
- `data/`：包含訓練及驗證資料集
- `model/`：儲存訓練好的模型

## 系統需求
- Python 3.8 或以上
- PyTorch
- Streamlit

## 安裝依賴
執行以下指令安裝所需套件：
```bash
pip install -r requirements.txt
```

## 注意事項
- 請確保 `data/` 資料夾中包含正確的訓練及驗證資料集。
- 模型訓練可能需要 GPU 加速，請確認系統是否支援 CUDA 或 Metal (MPS)。

## 授權
此專案採用 MIT 授權。
