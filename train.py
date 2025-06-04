# train.py
import os
import platform
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

#這邊需要注意 不適用生產環境
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = 'data/processed'
MODEL_PATH = 'model/food_model.pth'
CLASS_INDEX_PATH = 'model/class_to_idx.json'
EPOCHS = 5

# --- 自動偵測裝置 ---
def get_best_device():
    system = platform.system()
    processor = platform.processor().lower()

    if torch.backends.mps.is_available():
        print("🔧 檢測到 Apple Silicon，啟用 Metal (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"🔧 檢測到 NVIDIA GPU：{torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("🔧 未偵測到 GPU，使用 CPU 執行")
        return torch.device("cpu")


# --- 主訓練函式 ---
def train():
    device = get_best_device()

    # 根據裝置調整 BATCH_SIZE
    if device.type == "cuda":
        BATCH_SIZE = 32
    elif device.type == "mps":
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 8

    print(f"✅ 使用裝置：{device}，Batch size：{BATCH_SIZE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    num_classes = len(train_dataset.classes)

    # 建立模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練流程
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"訓練 Epoch {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"🧠 Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

        # 驗證流程
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="驗證中"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"✅ 驗證準確率: {acc:.2f}%")

    # 儲存模型與類別索引
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASS_INDEX_PATH, 'w') as f:
        json.dump(train_dataset.class_to_idx, f)

    print(f"\n🎉 訓練完成，模型儲存於：{MODEL_PATH}")
    print(f"📦 類別索引儲存於：{CLASS_INDEX_PATH}")


if __name__ == '__main__':
    train()
