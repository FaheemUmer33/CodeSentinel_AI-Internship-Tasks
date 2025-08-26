# 🐱🐶 Task 4: Image Classifier with Transfer Learning (Cats vs Dogs)

## 📋 Project Overview
This project implements an **image classification model** to distinguish between **cats** and **dogs** using **Transfer Learning** with **MobileNetV2**.  
The model is fine-tuned on the **Cats vs Dogs dataset** and deployed using a **Gradio web app** for interactive predictions.

---

## 📁 Dataset
- **Source:** [Kaggle – Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)  
- **Description:**  
  The dataset contains **25,000 labeled images** of cats and dogs.  
  It was split into training, validation, and test sets for model development.

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Image resizing to `224x224`
   - Normalization (`0–1` scale)
   - Data augmentation (rotation, flip, zoom, etc.)

2. **Model Building**
   - Base model: **MobileNetV2** (pretrained on ImageNet)
   - Added custom head: GlobalAveragePooling + Dense layers
   - Step 1: Train only head layers (feature extraction)
   - Step 2: Fine-tune last layers of base model (transfer learning)

3. **Training**
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Metrics: Accuracy
   - Early stopping & checkpointing used

4. **Evaluation**
   - Training vs Validation Accuracy/Loss curves
   - Final test accuracy reported
   - Confusion matrix for detailed performance

5. **Deployment**
   - Final model saved as `transfer_mobilenetv2_final.h5`
   - CSV logs and best checkpoint stored
   - Gradio app built for easy predictions with **example images**

---

## 🚀 Gradio Web App
The model is deployed with a **Gradio interface**.  

Features:
- Upload an image (cat/dog) for prediction
- View predicted probabilities for each class
- See the final prediction with confidence
- Try with **preloaded example images** for instant demo

```python
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load saved model
model = load_model("transfer_mobilenetv2_final.h5")

# Class labels
class_names = ["Cat", "Dog"]

# Preprocess
def preprocess(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction
def predict(img):
    img_array = preprocess(img)
    preds = model.predict(img_array)
    if preds.shape[1] == 1:  # sigmoid
        prob = float(preds[0][0])
        probs = {class_names[0]: 1 - prob, class_names[1]: prob}
        pred_class = class_names[1] if prob > 0.5 else class_names[0]
        confidence = prob if prob > 0.5 else 1 - prob
    else:  # softmax
        probs = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}
        pred_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))
    return probs, f"✅ Prediction: {pred_class} ({confidence:.2%})"

# Launch
demo = gr.Interface(fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=2), gr.Textbox()],
    title="🐶🐱 Cat vs Dog Classifier (MobileNetV2)",
    description="Upload an image or try example images for instant predictions."
)
demo.launch()

📊 Results

Final Model Accuracy: ~95% (after fine-tuning)

Robust to variations in lighting, pose, and background

Fast inference due to MobileNetV2 lightweight architecture

📦 Artifacts

transfer_mobilenetv2_final.h5 → Final trained model

training_log.csv → Training logs

best_model.h5 → Best checkpointed model

Gradio App → Interactive demo

🎥 Demo Video

A walkthrough video demonstrates:

Data preparation

Model training & fine-tuning

Evaluation results

Gradio app with predictions