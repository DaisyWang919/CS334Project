import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('data_selected/base_selected_train.csv')
test_data = pd.read_csv('data_selected/base_selected_test.csv')
X_train = train_data.drop('fraud_bool', axis=1)
y_train = train_data['fraud_bool']
X_test = test_data.drop('fraud_bool', axis=1)
y_test = test_data['fraud_bool']


mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 neurons in the first and 50 in the second
    max_iter=5,
    alpha=0.001,
    solver='adam',
    verbose=10,
    random_state=1,
    learning_rate_init=.001
)


mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

precision = precision_score(y_test, y_pred)
print(f"Model Precision: {precision}")

recall = recall_score(y_test, y_pred)
print(f"Model Recall: {recall}")

f1 = f1_score(y_test, y_pred,average= 'weighted')
print(f"Model F1 Score: {f1}")

y_probs = mlp.predict_proba(X_test)[:, 1]  
auc_roc = roc_auc_score(y_test, y_probs)
print(f"Model AUC-ROC: {auc_roc}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
