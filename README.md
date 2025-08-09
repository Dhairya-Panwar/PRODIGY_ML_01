# PRODIGY_ML_01
Machine Learning project for predicting house prices using Linear Regression, created as a learning exercise.


# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview
This project implements a **Linear Regression** model to predict house prices based on:
- Square footage
- Number of bedrooms
- Number of bathrooms

It was developed as a **learning exercise** to understand regression modeling, model evaluation, and creating a simple ML-powered web application using **Streamlit**.

---

## 📂 Dataset
**Source:** [Ames Housing Dataset - Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

**Files Used:**
- `training.csv` → Data for training the model  
- `testing.csv` → Data for making predictions  
- `sample_submission.csv` → Template for submission format  
- `data_description.txt` → Feature descriptions

---

## 🛠 Features Used in the Model
- **GrLivArea** → Above-ground living area (square feet)  
- **BedroomAbvGr** → Bedrooms above grade  
- **FullBath** → Full bathrooms above grade  
- **HalfBath** → Half bathrooms above grade  

---

## 🧰 Tech Stack
- **Python**  
- **Pandas**, **NumPy** → Data handling  
- **scikit-learn** → Machine learning model  
- **Streamlit** → Web application interface  

---

## 📈 Model Performance
- **R² Score:** ~0.63  
- **RMSE:** ~53,018  
> Note: Model performance is limited due to the intentionally small feature set, as per assignment requirements.

---
