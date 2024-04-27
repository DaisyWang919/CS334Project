import pandas as pd

# Data for neural network performance
nn_performance = {
    'Accuracy': [0.9893, 0.9888, 0.9889, 0.9889, 0.9891],
    'F1 Score': [0.9842, 0.9834, 0.9836, 0.9836, 0.9838],
    'ROC AUC': [0.8857, 0.8785, 0.8910, 0.8910, 0.8786]
}

# Data for logistic regression performance
lr_performance = {
    'Accuracy': [0.9893, 0.9888, 0.9889, 0.9889, 0.9891],
    'F1 Score': [0.9841, 0.9834, 0.9836, 0.9836, 0.9838],
    'ROC AUC': [0.8743, 0.8601, 0.8834, 0.8834, 0.8684]
}

# Variants
variants = ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV']

# Create DataFrames
nn_df = pd.DataFrame(nn_performance, index=variants)
lr_df = pd.DataFrame(lr_performance, index=variants)

# Combine DataFrames for display
combined_df = pd.concat([nn_df.add_prefix('NN '), lr_df.add_prefix('LR ')], axis=1)
combined_df