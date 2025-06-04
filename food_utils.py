import json
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# 載入模型與標籤對應
with open('model/class_to_idx.json', 'r') as f:
    CLASS_NAMES = list(json.load(f).keys())

def load_model(model_path, num_classes=101):  # 更新 num_classes 為 101
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_idx = torch.argmax(probs).item()
        if top_idx >= len(CLASS_NAMES):
            raise ValueError(f"模型輸出的索引 {top_idx} 超出 CLASS_NAMES 的範圍")
        return CLASS_NAMES[top_idx], probs[top_idx].item()

def get_calories(food_name, lookup_path='calories_lookup.json'):
    with open(lookup_path, 'r') as f:
        data = json.load(f)
    return data.get(food_name, '未知')