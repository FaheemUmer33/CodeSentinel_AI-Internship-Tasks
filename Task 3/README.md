# 🖼️ Task 3: Image Classification using the MNIST Dataset

## 📋 Project Overview
This project implements an **image classification model** to recognize handwritten digits (0–9) using the **MNIST dataset**.  
The goal is to build a deep learning model that can accurately classify digit images and provide an interactive demo using **Gradio**.

---

## 📁 Dataset
- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- The dataset contains **60,000 training images** and **10,000 testing images** of handwritten digits.  
- Each image is **28x28 pixels**, grayscale.

---

## 🛠️ Implementation Steps
1. **Data Preprocessing**
   - Normalize pixel values (0–255 → 0–1).  
   - Reshape images for model input.  
   - One-hot encode labels.  

2. **Model Building**
   - Built a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras**.  
   - Layers: Convolution, MaxPooling, Flatten, Dense, Dropout.  
   - Compiled with **Adam optimizer** and **categorical crossentropy loss**.  

3. **Model Training**
   - Trained on the MNIST training set.  
   - Validated on the test set.  
   - Evaluated accuracy and loss.  

4. **Deployment with Gradio**
   - Developed a **Gradio web app** for live predictions.  
   - Users can:
     - Upload an image of a digit.  
     - Draw a digit on a canvas.  
   - Model predicts the digit (0–9).  

---

## 📊 Results
- Achieved **high accuracy (>98%)** on the test dataset.  
- The Gradio app provides **real-time predictions** with visualization.  

---

## 🚀 How to Run
1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <your-repo-link>
   cd Task-3-MNIST-Image-Classification




Install dependencies:

pip install tensorflow gradio numpy matplotlib


Run the app:

python app.py


Open the Gradio link in your browser to test the model.

📹 Demo

A short video demo is provided in the submission showing:

Model training process.

Running the Gradio web app.

Uploading & drawing digits for prediction.

🔑 Key Learnings

Practical understanding of CNNs for image classification.

Importance of data preprocessing in improving accuracy.

Deploying ML models with interactive web apps.