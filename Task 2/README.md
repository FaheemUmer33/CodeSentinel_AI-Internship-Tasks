# 🏷️ Task 2: Customer Segmentation using K-Means

## 📋 Project Overview
This project implements **Customer Segmentation** using the **Mall Customer Dataset**.  
The goal is to group customers into distinct clusters based on their characteristics (e.g., Age, Income, Spending Score) using **K-Means Clustering**.  
This segmentation helps businesses understand their customers better and design targeted marketing strategies.

---

## 📁 Dataset
- **Source**: [Kaggle - Mall Customer Dataset](https://www.kaggle.com/vetrirah/customer)  
- **File Used**: `Train.csv`

**Features:**
- `CustomerID`: Unique customer ID  
- `Gender`: Male/Female  
- `Age`: Age of customer  
- `Annual Income (k$)`: Income of customer  
- `Spending Score (1-100)`: Spending behavior  

---

## ⚙️ Steps Performed
1. **Data Loading & Cleaning**
   - Imported dataset using pandas.
   - Checked for missing values and duplicates.
   - Encoded categorical features (`Gender`).

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions of age, gender, income, and spending score.
   - Scatterplots and pairplots for relationships between features.

3. **Feature Scaling**
   - Standardized features for better clustering performance.

4. **Optimal Clusters Selection**
   - Used **Elbow Method** and **Silhouette Score** to determine the best `k`.

5. **K-Means Clustering**
   - Applied K-Means to group customers into meaningful clusters.

6. **Visualization**
   - 2D & 3D scatter plots of clusters.
   - Heatmaps of cluster centroids.

7. **Model Saving**
   - Best clustering model saved using `joblib`.

---

## 📊 Results
- Identified **clear customer groups**:
  - High Income - High Spending  
  - High Income - Low Spending  
  - Low Income - High Spending  
  - Average Customers  
- Visualization showed distinct separations between clusters.  

---

## 🚀 Gradio App
An interactive **Gradio Interface** is provided to:
- Upload customer details (Age, Gender, Income, Spending Score).
- Get the **predicted cluster**.
- Explore **cluster visualizations** (scatter plots, heatmaps, and comparisons).

Run the app:
```bash
python app.py

or in Jupyter/Colab:

demo.launch()

📌 Tech Stack

Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Gradio, Joblib