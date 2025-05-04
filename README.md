# Finance-ML-Suite
# 📊 Financial Sentiment & Price Prediction App

## 🎓 Course Name:
**Programming for Finance**

## 👨‍🏫 Instructor:
**Usama Janjua**

## 🧠 App Overview

This Streamlit web application enables users to analyze and predict the prices of cryptocurrencies based on sentiment analysis data. Developed as part of the Programming for Finance course, it leverages machine learning models to assess how sentiment scores (e.g., from social media or news) influence cryptocurrency prices.

### 🔍 Key Features

- **File Upload:** Users can upload their own CSV datasets.
- **Data Display:** View uploaded data in a clean, interactive table.
- **Data Visualization:** Line charts for both sentiment scores and prices.
- **Machine Learning Models:**
  - Linear Regression
  - Random Forest Regressor
- **Model Training and Evaluation:**
  - R² Score and Mean Squared Error (MSE)
  - User-selectable feature columns
- **Price Prediction:**
  - Input custom sentiment scores for predictions
- **Downloadable Results:** Option to download trained models and prediction output.

### 📁 Expected CSV Format

The uploaded dataset should at minimum contain:
- A **'sentiment'** column (numeric sentiment scores)
- A **'price'** column (target variable for prediction)
- Optional: Additional numerical features for model training

### 🛠 How to Run Locally

1. **Clone the repository or copy the app files**
2. **Install dependencies:**

```bash
pip install -r requirements.txt

## Deployment Link:
👉   Local URL: http://localhost:8501
  Network URL: http://192.168.18.149:8501
