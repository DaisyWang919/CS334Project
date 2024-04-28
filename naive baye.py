import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # or use MultinomialNB, BernoulliNB based on your data
from sklearn.metrics import accuracy_score
train_data = pd.read_csv('../data_selected/base_selected_train.csv')
test_data = pd.read_csv('../data_selected/base_selected_test.csv')
X_train = train_data.drop('fraud_bool', axis=1)
y_train = train_data['fraud_bool']
X_test = test_data.drop('fraud_bool', axis=1)
y_test = test_data['fraud_bool']
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import joblib

# Instantiate the classifier
gnb = GaussianNB()

# Train the classifier
model = gnb.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Model Precision: {precision}")

# Calculate recall
recall = recall_score(y_test, y_pred)
print(f"Model Recall: {recall}")

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)
print(f"Model F1 Score: {f1}")

# Calculate probabilities for AUC
y_probs = model.predict_proba(X_test)[:, 1]  # get the probabilities of the positive class

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test, y_probs)
print(f"Model AUC-ROC: {auc_roc}")
joblib.dump(model, 'nb_model.pkl')