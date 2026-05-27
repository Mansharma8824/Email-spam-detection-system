import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score,recall_score, confusion_matrix,classification_report

# Load the DataSet
df = pd.read_csv('../Data/processed/cleaned_data.csv')
df.dropna(inplace=True)
X = df['processed_Masseges']
y = df['target']

# train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating ML pipelines 
Pipelines = {
    'Logistic Regression' :Pipeline([
        ('tfidf',TfidfVectorizer()),
        ('model',LogisticRegression())
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf',TfidfVectorizer()),
        ('model',MultinomialNB())
    ]),
    'SVC':Pipeline([
        ('tfidf',TfidfVectorizer()),
        ('model',LinearSVC())
    ]),
    'Random Forest': Pipeline([
        ('tfidf',TfidfVectorizer()),
        ('model',RandomForestClassifier())
    ])
}

# model parameters
params = {
'Logistic Regression': {
        'model__C': [0.1, 1, 10],
        'tfidf__max_features': [3000, 5000]
    },

    'Naive Bayes': {
        'model__alpha': [0.1, 0.5, 1.0],
        'tfidf__max_features': [3000, 5000]
    },

    'SVC': {
        'model__C': [0.1, 1, 10],
        'tfidf__max_features': [3000, 5000]
    },

    'Random Forest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10, 15],
        'tfidf__max_features': [3000, 5000]
    }
    
}

results ={}
from sklearn.model_selection import GridSearchCV
for model_name in Pipelines:
    print(f"Training the {model_name}")
    
    gsCV = GridSearchCV(
        Pipelines[model_name],
        param_grid= params[model_name],
        cv=5,
        scoring='f1'
    )
    gsCV.fit(X_train,y_train)
    
    y_pred = gsCV.predict(X_test)
    
    accurecy =  accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall =  recall_score(y_test,y_pred)
    f1_value = f1_score(y_test,y_pred)
      
    results[model_name] = {
        'best_params': gsCV.best_params_,
        'cv_f1':gsCV.best_score_,
        'best_estimator': gsCV.best_estimator_,
        "Accuracy": accurecy,
        "Precision": precision,
        "Recall":recall,
        "F1-Score": f1_value
    }
    
df_results = pd.DataFrame(results).T   # T = transpose for clean view
result_df = df_results.sort_values(by='F1-Score',ascending=False)
best_model = result_df.iloc[0].best_estimator

# save model
joblib.dump(best_model,'../app/models/model.pkl')
