# 🌸 Task 2: Iris Flower Classification

## 📋 Project Overview
This project implements a **Classification Model** using the **Iris dataset**, one of the most well-known datasets in machine learning.  
The goal is to classify iris flowers into three species — *Setosa, Versicolor, and Virginica* — based on their features (*sepal length, sepal width, petal length, petal width*).  

Multiple machine learning models were trained, evaluated, and compared to identify the best-performing classifier.

---

## 📁 Dataset
- **Source**: [Iris Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris)  
- **Features**:
  - `sepal length (cm)`  
  - `sepal width (cm)`  
  - `petal length (cm)`  
  - `petal width (cm)`  
- **Target**: Species (*Setosa, Versicolor, Virginica*)

---

## ⚙️ Steps Performed
1. **Data Loading & Exploration**
   - Loaded Iris dataset using scikit-learn & pandas.
   - Checked dataset shape, unique classes, and summary statistics.
   - Visualized feature distributions with histograms and pair plots.

2. **Data Preprocessing**
   - Handled missing values (if any).
   - Encoded target labels.
   - Scaled features using StandardScaler.

3. **Model Training**
   - Implemented multiple classification algorithms:
     - Logistic Regression  
     - Decision Tree Classifier  
     - K-Nearest Neighbors (KNN)

4. **Model Evaluation**
   - Compared models using accuracy scores.
   - Generated **confusion matrix** & **classification reports**.
   - Visualized decision boundaries for better interpretability.

5. **Model Selection**
   - Identified the best-performing model based on accuracy and evaluation metrics.

---

## 📊 Results
- Achieved high classification accuracy (>90%) across models.  
- Decision boundaries showed **clear separation of species**.  
- Classification report highlighted precision, recall, and F1-score for each class.  

---

## 🚀 Deliverables
- **Jupyter Notebook** with full code and model comparison.
- **Visualizations**: pair plots, confusion matrices, and decision boundaries.
- **Saved Models** for future use (joblib).

---

## 📌 Tech Stack
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  

---
