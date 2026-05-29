# 📧 Email Spam Detection System

![Python](https://img.shields.io/badge/Python-3.12.0-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A machine learning web application that detects whether an email is **spam or not spam** using NLP techniques and multiple ML models. Built with Python, Scikit-learn, and Flask.

---

## 🚀 Live Demo
> 🔗 Demo Link - https://email-spam-detection-system-1.onrender.com
> I am using free tair of Render because of that it can take few minutes for loading on browser
---

## 📌 Project Overview

This project builds an end-to-end ML pipeline that:
- Compares **4 machine learning models** to find the best one
- Uses **TF-IDF Vectorization** to convert text to numbers
- Uses **GridSearchCV** for hyperparameter tuning
- Serves predictions via a **Flask web app**

---

## 🎯 Model Performance

| Model | CV F1 Score | Test Accuracy |
|-------|-------------|---------------|
| Logistic Regression |  |  |
| Naive Bayes |  |  |
| SVC (LinearSVC) | |  |
| Random Forest | | |
| **✅ Best Model** | ** ** | ** ** |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.8+ |
| ML Library | Scikit-learn |
| NLP | TF-IDF Vectorizer |
| Web Framework | Flask |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
Email-Spam-Detection-System/
│
├── app.py                  ← Flask web application
├── train.py                ← ML pipeline training script
├── requirements.txt        ← All dependencies
├── README.md
│
├── data/
│   └── raw/                ← Raw dataset (CSV)
│
├── models/
│   └── best_model.pkl      ← Saved best model (auto-generated)
│
├── templates/
│   └── index.html          ← Frontend UI
│
├── static/
│   └── style.css           ← Styling
│
└── notebooks/
    └── EDA_experiments.ipynb  ← Research & experiments
```

---

## ⚙️ ML Pipeline

```
Raw Email Text
      ↓
TF-IDF Vectorizer       ← converts text to numerical features
      ↓
ML Model                ← Logistic Regression / NB / SVC / Random Forest
      ↓
GridSearchCV            ← finds best hyperparameters (cv=5)
      ↓
Best Model Saved        ← as best_model.pkl
      ↓
Flask App               ← serves predictions via web UI
```

---

## 📊 Dataset

- **Source:** [Kaggle — Email Spam Detection](https://www.kaggle.com/datasets/zeeshanyounas001/email-spam-detection)
- **Type:** Binary Classification (Spam / Not Spam)
- **Features:** Raw email message text

---

## ▶️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/Email-Spam-Detection-System.git
cd Email-Spam-Detection-System
```

### 2. Create and activate virtual environment
```bash
# Create
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (run once)
```bash
python train.py
```
This will train all 4 models, compare them, and save the best one as `models/best_model.pkl`

### 5. Start the Flask app
```bash
python app.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## 🖥️ App Screenshot

> _Add screenshot here after building the UI_

---

## 📦 Requirements

```
flask
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔍 Key Features

- **Pipeline architecture** — TF-IDF + Model in single pipeline (prevents data leakage)
- **4 models compared** — picks the best one automatically
- **Hyperparameter tuning** — GridSearchCV with 5-fold cross validation
- **Imbalanced data handled** — `class_weight='balanced'` used
- **F1 Score based selection** — better metric than accuracy for spam data
- **One-click prediction** — type any email text and get instant result

---

## 👤 Author

**Manish Kumar**
- GitHub: [@Mansharma8824](https://github.com/Mansharma8824)
- LinkedIn: ([Manish Kumar](https://www.linkedin.com/in/manish-kumar-2518b6242/))

---
