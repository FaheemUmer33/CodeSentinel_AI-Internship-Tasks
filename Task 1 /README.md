# 🏠 Task 1: Linear Regression — House Price Prediction

## 📋 Project Overview
This project implements a **Linear Regression model** to predict house prices using the **California Housing dataset** (from scikit-learn).  
The model is trained on one or two features (e.g., *average rooms per household* and *median income*) to demonstrate simple regression concepts.  
It includes **model training, evaluation, visualization, and an interactive Gradio app interface**.

---

## 📁 Dataset
- **Source**: [California Housing Dataset (scikit-learn)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)  
- **Target**: Median house value (in $100,000s)  
- **Features Used**:
  - `AveRooms` → Average number of rooms per household  
  - `MedInc` → Median income in block group  

---

## ⚙️ Tech Stack
- **Language**: Python  
- **Libraries**:  
  - `scikit-learn` → model training & evaluation  
  - `pandas`, `numpy` → data handling  
  - `matplotlib` → visualization  
  - `gradio` → interactive app interface  
  - `joblib` → model persistence  

---

## 🚀 Implementation Steps
1. **Data Loading**: Load California Housing dataset from scikit-learn.  
2. **Feature Selection**: Choose one (`AveRooms`) or two (`AveRooms`, `MedInc`) features.  
3. **Train/Test Split**: Split dataset into 80% training and 20% testing.  
4. **Model Training**: Train `LinearRegression` model using scikit-learn.  
5. **Evaluation**: Compute MAE, RMSE, R² metrics.  
6. **Visualization**:  
   - Regression line for one-feature model  
   - Actual vs Predicted scatter plot  
   - Residuals plot  
   - Optional 3D regression plane  
7. **Gradio Interface**: Build a simple app for real-time predictions.  
8. **Model Saving**: Save trained model using `joblib` for reuse.  

---

## 📊 Results
- **One-Feature Model (AveRooms)**: Provided baseline performance.  
- **Two-Feature Model (AveRooms + MedInc)**: Showed improved accuracy (higher R², lower RMSE).  
- **Visualization** clearly demonstrated linear trends and residual errors.  

---

## 🌐 Gradio App
An interactive **Gradio web app** is provided:  
- Input:  
  - Average Rooms per Household  
  - Median Income (10k$ scale)  
- Output: Predicted House Price in USD  

In Google Colab, running the app cell provides a **public link** to interact with the model.  

---

## 📦 How to Run
1. Clone the repo / open the Colab notebook.  
2. Install dependencies (if needed):  
   ```bash
   pip install scikit-learn pandas matplotlib joblib gradio

Run all notebook cells in order.

Launch the Gradio app from the last cell to test predictions interactively.

📌 Submission Checklist

 Data loaded and features selected

 Train/test split applied

 Model trained (Linear Regression)

 Metrics reported (MAE, RMSE, R²)

 Plots created (line, scatter, residuals)

 Gradio app implemented

 Model saved using joblib