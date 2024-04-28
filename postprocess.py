import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
import pandas as pd


df = pd.read_csv('../data_selected/base_selected_test.csv')
X_test = df.drop('fraud_bool', axis=1)
y_test = df['fraud_bool']



model_display_names = {
    'decision_tree__model.pkl': 'Decision Tree',
    'knn_model.pkl': 'K-Nearest Neighbors',
    'logistic_regression_model.pkl': 'Logistic Regression',
    'nb_model.pkl': 'Naive Bayes',
    'nn_model.pkl': 'Neural Network'
}


plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label="Random chance")


for model_file in model_display_names:
    model_path = f'{model_file}'
    model = joblib.load(model_path)
    
    y_scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    roc_auc = roc_auc_score(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    
    model_name_for_legend = model_display_names[model_file]
    plt.plot(fpr, tpr, label=f'{model_name_for_legend} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()