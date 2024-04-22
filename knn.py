import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, 1:].values  # All columns except the first
    y = df.iloc[:, 0].values   # First column as target
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_knn(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    print("Best KNN parameters:", grid_search.best_params_)
    return best_knn

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:,1]
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr")
    fpr, tpr, _ = roc_curve(y_test, probabilities.ravel())
    return accuracy, f1, roc_auc

def save_model(model):
    joblib.dump(model, 'logistic_regression_model.pkl')

if __name__ == "__main__":
    X, y = load_data("data_selected/base_selected.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    best_knn = train_knn(X_train, y_train)
    accuracy, f1, roc_auc = evaluate_model(best_knn, X_test, y_test)
    print(f"KNN Test Accuracy: {accuracy}")
    print(f"KNN Test F1 Score: {f1}")
    print(f"KNN Test ROC AUC: {roc_auc}")
    save_model(best_knn)