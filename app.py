# app.py
import streamlit as st
from PIL import Image
from food_utils import load_model, predict_image, get_calories

st.title("🍱 FoodVision Lite")

uploaded_file = st.file_uploader("請上傳食物圖片", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='你上傳的圖片', use_container_width=True)

    model = load_model("model/food_model.pth")
    label, confidence = predict_image(image, model)
    calories = get_calories(label)

    st.markdown(f"### 🥘 辨識結果：{label}")
    st.markdown(f"信心分數：{confidence:.2f}")
    st.markdown(f"預估熱量：{calories} kcal/100g")
