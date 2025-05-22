# 🎓 Student Performance Predictor

A web application built with **Streamlit** that predicts a student's math score based on various academic and demographic features. The model is trained using multiple models, and the app allows users to interactively test predictions using their own inputs.

---

## 📌 Features

- Predict student's **Math Score** based on:
  - Gender
  - Race/Ethnicity
  - Parental Level of Education
  - Lunch type
  - Test preparation course
  - Reading and Writing scores
- Interactive user interface built using **Streamlit**
- Real-time predictions with a trained **Linear Regression model**
- Uses `joblib` for loading model and preprocessing pipeline

---

## 🧠 Model

- Model: Linear Regression(best_model)
- Preprocessing: One-hot encoding for categorical features, standard scaling for numerical ones
- Files:
  - `linear_regression_model.pkl`
  - `preprocessor.pkl`

---

