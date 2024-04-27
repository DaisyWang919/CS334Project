import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

# Load the saved logistic regression model
model = joblib.load('decision_tree__model.pkl')
# model = joblib.load('nn_model.pkl')
# Filenames of the test datasets
variant_test_files = [
    'base_selected_test.csv',
    'variantI_selected_test.csv',
    'variantII_selected_test.csv',
    'variantIII_selected_test.csv',
    'variantIV_selected_test.csv'
]

# Directory where the test files are located
data_directory = "../data_selected/"

results = ""

# Test the model on each variant's test set and accumulate the results in the string
for file in variant_test_files:
    # Load the test data
    test_data = pd.read_csv(data_directory + file)
    X_test = test_data.drop('fraud_bool', axis=1)  # No need to convert to values; we handle this next
    y_test = test_data['fraud_bool']
    
    # Make predictions on the test data using the feature names to match the model's fitting
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probabilities of the positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions.round())
    f1 = f1_score(y_test, predictions.round(), average='weighted')
    roc_auc = roc_auc_score(y_test, probabilities)
    
    # Append each variant's results to the results string
    results += f"{file:<30} {accuracy:<10.4f} {f1:<10.4f} {roc_auc:<10.4f}\n"

# Print the header
print(f"{'Variant':<30} {'Accuracy':<10} {'F1 Score':<10} {'ROC AUC':<10}")
print('-' * 60)

# Print the accumulated results
print(results)