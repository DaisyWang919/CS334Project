from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pandas as pd
import joblib

# Load dataset
print("Loading data...")
df = pd.read_csv('../data_selected/base_selected_train.csv')
X = df.drop('fraud_bool', axis=1)
y = df['fraud_bool']

# Split dataset into training and test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define pipeline and parameter grid
print("Setting up pipeline and parameter grid...")
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 10],
    'classifier__min_samples_leaf': [1, 4]
}

# Hyperparameter tuning with cross-validation
print("Starting GridSearchCV...")
clf = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=2)
clf.fit(X_train, y_train)

# Evaluate the best model
print("Evaluating the best model...")
best_model = clf.best_estimator_
y_pred = best_model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_pred)

print(f"Best Decision Tree parameters: {clf.best_params_}")
print(f"Decision Tree Test F1 Score: {test_f1_score:.4f}")
print(f"Decision Tree Test ROC AUC: {test_roc_auc:.4f}")

joblib.dump(best_model, 'decision_tree__model.pkl')