import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd

# Load the saved logistic regression model
model = joblib.load('logistic_regression_model.pkl')

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

# Print the header of the table
print(f"{'Variant':<15} {'Accuracy':<10} {'F1 Score':<10} {'ROC AUC':<10}")

# Test the model on each variant's test set and store the results
for file in variant_test_files:
    # Load the test data
    test_data = pd.read_csv(data_directory + file)
    X_test = test_data.drop('fraud_bool', axis=1).values  # Use values to avoid passing DataFrame with column names
    y_test = test_data['fraud_bool'].values
    
    # Make predictions on the test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probabilities of the positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions,average='weighted')
    roc_auc = roc_auc_score(y_test, probabilities)
    
    # Print each variant's results in a formatted row
    print(f"{file:<15} {accuracy:<10.4f} {f1:<10.4f} {roc_auc:<10.4f}")