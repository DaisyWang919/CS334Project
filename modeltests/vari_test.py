import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd


model = joblib.load('decision_tree__model.pkl')

variant_test_files = [
    'base_selected_test.csv',
    'variantI_selected_test.csv',
    'variantII_selected_test.csv',
    'variantIII_selected_test.csv',
    'variantIV_selected_test.csv'
]


data_directory = "../data_selected/"

results = ""

for file in variant_test_files:

    test_data = pd.read_csv(data_directory + file)
    X_test = test_data.drop('fraud_bool', axis=1)  
    y_test = test_data['fraud_bool']
    

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  

    accuracy = accuracy_score(y_test, predictions.round())
    f1 = f1_score(y_test, predictions.round(), average='weighted')
    roc_auc = roc_auc_score(y_test, probabilities)

    results += f"{file:<30} {accuracy:<10.4f} {f1:<10.4f} {roc_auc:<10.4f}\n"


print(f"{'Variant':<30} {'Accuracy':<10} {'F1 Score':<10} {'ROC AUC':<10}")
print('-' * 60)


print(results)