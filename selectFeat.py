import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def cal_corr(df):
    return df.corr()


def select_features(trainDF, testDF):
    continuous_vars = ['income', 'name_email_similarity', 'current_address_months_count', 
                   'customer_age', 'days_since_request', 'intended_balcon_amount', 
                   'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 
                   'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 
                   'credit_risk_score', 'bank_months_count', 'proposed_credit_limit', 
                   'session_length_in_minutes']
    
    corr = trainDF.iloc[1:, 1:].corr()
    corr_with_target = trainDF.corr().iloc[1:, 0]
    selected_features = continuous_vars.copy()
    deleted = []
    for feature in continuous_vars:
        if feature in selected_features:
            if corr_with_target[feature] < -0.28:
                selected_features.remove(feature)
                deleted.append(feature)
            correlated_features = corr[(corr[feature] < -0.25) | (corr[feature] > 0.25)].index.tolist()
            correlated_features.remove(feature)
            for correlated_feature in correlated_features:
                if correlated_feature in selected_features:
                    selected_features.remove(correlated_feature)
                    deleted.append(correlated_feature)
    
    selected_features.extend([col for col in trainDF.columns if col not in continuous_vars])
    
    print("Deleted features:", deleted)
    return deleted

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    continuous_vars = ['income', 'name_email_similarity', 'current_address_months_count', 
                   'customer_age', 'days_since_request', 'intended_balcon_amount', 
                   'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 
                   'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 
                   'credit_risk_score', 'bank_months_count', 'proposed_credit_limit', 
                   'session_length_in_minutes']
    df = pd.read_csv("data/variantV_cleaned.csv")
    corr = cal_corr(df[continuous_vars])
    plt.figure(figsize=(13, 13))
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=.5)
    plt.title("Continuous Variable Correlation")
    plt.show()
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    print("Original Shape:", df.shape)
    # calculate the training correlation
    deleted = select_features(train, test)
    train = train.drop(deleted, axis=1)
    test = test.drop(deleted, axis=1)
    print("Transformed Training Shape:", train.shape)
    # save it to csv
    train.to_csv("data_selected/variantV_selected_train.csv", index=False)
    test.to_csv("data_selected/variantV_selected_test.csv", index=False)


if __name__ == "__main__":
    main()



