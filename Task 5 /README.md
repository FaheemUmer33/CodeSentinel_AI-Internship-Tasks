# 🐱🐶 Task 5: AI-Powered Web App with Streamlit (Cats vs Dogs Classifier)

## 📋 Project Overview
This project develops a **simple AI-powered web application** using **Streamlit**.  
It allows users to **upload an image** (cat or dog) and get a prediction from a **pre-trained model** built in **Task 4** using **Transfer Learning (MobileNetV2)**.  

The app provides a clean and interactive UI, making it easy for users to test the model in real-time.

---

## 🧑‍💻 Features
- 📂 **Upload an image** (JPG, JPEG, PNG)  
- ⚡ **Model prediction** using trained MobileNetV2 transfer learning model  
- 🎨 **Streamlit UI** with file uploader and prediction result display  
- 🌍 **Deploy locally or via ngrok/Hugging Face/Render** for sharing  

---

## 📁 Project Structure

📦 Task 5 - Streamlit Image Classifier
┣ 📜 streamlit_app.py # Main Streamlit app
┣ 📜 transfer_mobilenetv2_final.h5 # Trained model (from Task 4)
┗ 📜 README.md # Project documentation


---

## 🚀 Setup Instructions

### 1️⃣ Install Dependencies
```bash
pip install streamlit tensorflow pillow numpy pyngrok

2️⃣ Run the App Locally
streamlit run streamlit_app.py


The app will open in your browser at http://localhost:8501.

🌍 Run in Google Colab with ngrok

If running inside Google Colab:

!pip install streamlit pyngrok tensorflow pillow numpy
!ngrok config add-authtoken YOUR_NGROK_TOKEN


Then run:

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("Public URL:", public_url)

!streamlit run streamlit_app.py --server.port 8501 &> /dev/null &


👉 Use the Public URL to access the app online.

📊 Model Details

Base Model: MobileNetV2 (pre-trained on ImageNet)

Transfer Learning: Fine-tuned on Cats vs Dogs dataset

Accuracy: Optimized using early stopping, checkpointing, and data augmentation