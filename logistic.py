import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, 1:].values  # All columns except the first
    y = df.iloc[:, 0].values   # First column as target
    return X, y


def train_logistic_regression(X_train, y_train):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=200)
    grid_search = GridSearchCV(lr, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_lr = grid_search.best_estimator_
    print("Best Logistic Regression parameters:", grid_search.best_params_)
    return best_lr

def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr")
        fpr, tpr, _ = roc_curve(y_test, probabilities.ravel())
        auc_score = auc(fpr, tpr)
        return accuracy, f1, roc_auc, auc_score

if __name__ == "__main__":
    X_train, y_train = load_data("data_selected/base_selected_train.csv")
    X_test, y_test = load_data("data_selected/base_selected_test.csv")
    best_lr = train_logistic_regression(X_train, y_train)
    accuracy, f1, roc_auc, auc_score = evaluate_model(best_lr, X_test, y_test)
    print(f"Logistic Regression Test Accuracy: {accuracy}")
    print(f"Logistic Regression Test F1 Score: {f1}")
    print(f"Logistic Regression Test ROC AUC: {roc_auc}")
    print(f"Logistic Regression Test AUC: {auc_score}")
