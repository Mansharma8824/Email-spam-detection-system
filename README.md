# 📧 Email Spam Detection System

![Python](https://img.shields.io/badge/Python-3.12.0-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A end to end machine learning web application that detects whether an email is **spam or not spam** using NLP techniques and multiple ML models. Built with Python, Scikit-learn, and Flask.

---

## 🚀 Live Demo
> 🔗 Demo Link - https://email-spam-detection-system-1.onrender.com
- I am using free tair of Render because of that it can take few seconds for loading on browser
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
| Logistic Regression | 0.902655 | 0.978703 |
| Naive Bayes | 0.91453 | 0.980639 |
| SVC (LinearSVC) | 0.908297 | 0.979671 |
| Random Forest | 0.891892 | 0.976767 |
| **✅ Naive Bayes** | 0.91453 | 0.980639 |

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12+ |
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
├── data/
│   └── raw            
│         ├──mail.csv ← Raw dataset (CSV)
│   └── processed           
│         ├── cleaned_data.csv ← Cleaned dataset (CSV)
├── app/
│   └──models/
|          ├── model.pkl      ← Saved best model (auto-generated)
│   └──templates/
|          ├── index.html     ← Frontend UI
│   └── route.py     ← Flask web application
│   └── train.py   ← ML pipeline training script
│   └── utils.py   ← text preprocessing script (Common functions)
│
└── notebooks/
|    ├── mail_data_cleaning.ipynb  ← Data collection, Data cleaning, EDA, Data Preprocessing
|    ├── model_training.ipynb  ← Feature Engineering(Vectorization), Model Training
├── requirements.txt        ← All dependencies
├── README.md
├── .gitignore
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
Best Model Saved        ← as model.pkl
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
git clone https://github.com/Mansharma8824/Email-Spam-Detection-System.git
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
python model_train.py
```
This will train all 4 models, compare them, and save the best one as `models/model.pkl`

### 5. Start the Flask app
```bash
python route.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## 🖥️ App Screenshot

> <img width="1443" height="880" alt="Screenshot 2026-05-29 232619" src="https://github.com/user-attachments/assets/209e5087-1839-449b-b813-4d00c1781687" />
<img width="1458" height="858" alt="Screenshot 2026-05-29 232600" src="https://github.com/user-attachments/assets/d428ff05-a454-4a4b-9956-5e3df93325c9" />
<img width="1458" height="856" alt="Screenshot 2026-05-29 232512" src="https://github.com/user-attachments/assets/010c527b-a449-4d09-8b5a-5f3ff14f6b2c" />
<img width="1419" height="767" alt="Screenshot 2026-05-29 232456" src="https://github.com/user-attachments/assets/51e69e42-2f85-497b-8d77-ee23ff3b7b64" />
<img width="1544" height="814" alt="Screenshot 2026-05-29 232216" src="https://github.com/user-attachments/assets/c40c7c04-2972-46ef-9386-f50e25c559e0" />
<img width="1589" height="805" alt="Screenshot 2026-05-29 232055" src="https://github.com/user-attachments/assets/6a10ec47-cc44-47d4-abab-1f920ef8f13c" />


---

## 📦 Requirements

```
flask
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
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
- **F1 Score based selection** — better metric than accuracy for spam data
- **One-click prediction** — type any email text and get instant result

---

## 👤 Author

**Manish Kumar**
- GitHub: [@Mansharma8824](https://github.com/Mansharma8824)
- LinkedIn: ([Manish Kumar](https://www.linkedin.com/in/manish-kumar-2518b6242/))

---
