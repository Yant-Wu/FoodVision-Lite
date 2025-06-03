import os
import shutil

# 原始圖片資料夾（每個類別一個資料夾）
SOURCE_DIR = "food-101/images"

# 目標分類資料夾
TARGET_DIR = "data/processed"

# 訓練/驗證清單
TRAIN_LIST = "food-101/meta/train.txt"
VAL_LIST = "food-101/meta/test.txt"

def copy_images(list_path, split_name):
    with open(list_path, "r") as f:
        lines = f.read().strip().splitlines()

    print(f"📁 處理 {split_name} 集，共 {len(lines)} 張圖片")

    for line in lines:
        # line 格式範例：pizza/pizza_00123
        label, filename = line.split("/")
        src_file = os.path.join(SOURCE_DIR, label, f"{filename}.jpg")
        dest_dir = os.path.join(TARGET_DIR, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, f"{filename}.jpg")

        if not os.path.exists(src_file):
            print(f"❌ 找不到圖片：{src_file}")
            continue

        shutil.copy(src_file, dest_file)

    print(f"✅ 已完成 {split_name} 集分類，總數：{len(lines)} 張圖片\n")

if __name__ == "__main__":
    print("🚀 開始將 food-101 圖片分類到 train/val ...\n")

    # 建立資料夾
    os.makedirs(os.path.join(TARGET_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "val"), exist_ok=True)

    # 分類訓練與驗證資料
    copy_images(TRAIN_LIST, "train")
    copy_images(VAL_LIST, "val")

    print("🎉 所有圖片已分類完畢，可開始訓練模型！")
