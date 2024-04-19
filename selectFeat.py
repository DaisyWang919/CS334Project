import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def cal_corr(df):
    return df.corr()


def select_features(df):
    continuous_vars = ['income', 'name_email_similarity', 'current_address_months_count', 
                   'customer_age', 'days_since_request', 'intended_balcon_amount', 
                   'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 
                   'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 
                   'credit_risk_score', 'bank_months_count', 'proposed_credit_limit', 
                   'session_length_in_minutes']
    
    corr = df[continuous_vars].corr()
    corr_with_target = corr.iloc[:, 0]
    selected_features = continuous_vars.copy()
    
    for feature in continuous_vars:
        if feature in selected_features:
            if corr_with_target[feature] < -0.28:
                selected_features.remove(feature)
            correlated_features = corr[(corr[feature] < -0.15) | (corr[feature] > 0.15)].index.tolist()
            correlated_features.remove(feature)
            for correlated_feature in correlated_features:
                if correlated_feature in selected_features:
                    selected_features.remove(correlated_feature)
    
    selected_features.extend([col for col in df.columns if col not in continuous_vars])
    
    print("Selected features:", selected_features)
    df = df[[df.columns[0]] + selected_features]
    return selected_features


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
    df = pd.read_csv("data/base_cleaned.csv")
    print(df.columns)
    corr = cal_corr(df[continuous_vars])
    # plt.figure(figsize=(13, 13))
    # sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=.5)
    # plt.title("Continuous Variable Correlation")
    # plt.show()

    print("Original Shape:", df.shape)
    # calculate the training correlation
    selected_features = select_features(df)
    print("Transformed Training Shape:", df.shape)
    # save it to csv
    df.to_csv("data/base_selected.csv", index=False)
    df1 = pd.read_csv("data/variantI_cleaned.csv")
    df2 = pd.read_csv("data/variantII_cleaned.csv")
    df3 = pd.read_csv("data/variantII_cleaned.csv")
    df4 = pd.read_csv("data/variantIV_cleaned.csv")
    df5 = pd.read_csv("data/variantV_cleaned.csv")
    df1[selected_features].to_csv("data/variantI_selected.csv", index=False)
    df2[selected_features].to_csv("data/variantII_selected.csv", index=False)
    df3[selected_features].to_csv("data/variantIII_selected.csv", index=False)
    df4[selected_features].to_csv("data/variantIV_selected.csv", index=False)
    df5[selected_features].to_csv("data/variantV_selected.csv", index=False)



if __name__ == "__main__":
    main()



