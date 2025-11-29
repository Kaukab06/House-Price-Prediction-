#  House Price Prediction using Machine Learning

This project predicts **house sale prices** using property features such as area, rooms, location coordinates, and age.  
A fully deployed **Streamlit application** allows interactive prediction based on user inputs.

---

# Objective

✔ Understand impact of features on house prices  
✔ Compare multiple regression models  
✔ Build a real-world ML solution  
✔ Deploy on a web platform for instant predictions  

---

# Tech Stack & Features

| Component | Technology |
|----------|------------|
| Data Source | Kaggle / Sklearn Boston Housing Dataset |
| Algorithms | Linear Regression, RandomForest, XGBoost |
| Deployment | Streamlit |
| Preprocessing | Scaling & Encoding |
| Visualization | Seaborn, Matplotlib |
| Model Storage | Joblib Pickle |

---

# Repository Structure

 House_Price_Prediction/
│── data/ # Dataset (train.csv or boston.csv)
│── model_training.py # Training + evaluation + save model
│── app.py # Streamlit UI for prediction
│── saved_model.pkl # Final ML model
│── requirements.txt # Dependencies
│── README.md

yaml
Copy code

---

# Model Development Pipeline

1️ Load & clean dataset  
2️ Exploratory Data Analysis  
3️ Feature Engineering  
- Categorical Encoding  
- Outlier filtering  
- Log transform (optional)

4️ Training & model comparison  
5️ Cross-validation + tuning  
6️ Best model saved & deployed  

---

# Performance

| Model | R² Score | RMSE |
|------|---------|------|
| Linear Regression | ~0.80 | High variance |
| RandomForest (Best) | ~0.88–0.93 | Lower |
| XGBoost | Competitive | Tunable |

> Actual performance depends on dataset version used

---

# How to Run

```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
App opens here → http://localhost:8501/

 Streamlit App Preview
✔ User input sliders/text fields
✔ Instant predicted price output
✔ Helpful feature visualizations (optional)

Sample Output:

 Predicted Price: ₹ 75.3 Lakhs (example)

 Real-World Use Cases
Real estate companies

Investment decision tools

Urban analytics

Price negotiation assistants
