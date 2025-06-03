import json
import torch
from torchvision import transforms
from PIL import Image

# 載入模型與標籤對應
CLASS_NAMES = ['pizza', 'sushi', 'steak', 'burger', 'ramen', 'salad', 'fried_rice', 'pasta', 'cake', 'ice_cream']

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
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
        return CLASS_NAMES[top_idx], probs[top_idx].item()

def get_calories(food_name, lookup_path='calories_lookup.json'):
    with open(lookup_path, 'r') as f:
        data = json.load(f)
    return data.get(food_name, '未知')