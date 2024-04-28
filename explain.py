import shap
import joblib
import pandas as pd


model = joblib.load('nn_model.pkl')


df = pd.read_csv('../data_selected/base_selected_test.csv')
X_test = df.drop('fraud_bool', axis=1)
y_test = df['fraud_bool']
background = shap.sample(X_test, 100)
explainer = shap.KernelExplainer(model.predict,background)
test = shap.sample(X_test, 1000)
shap_values = explainer.shap_values(test)
shap.summary_plot(shap_values, test)
