import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def cal_corr(df):
    return df.corr()


def select_features(df):
    corr = df.iloc[:-1, 1:].corr()
    corr_with_target = df.corr().iloc[:-1, 0]
    selected_features = corr.columns.tolist()
    for feature in corr.columns:
        if feature in selected_features:
            if corr_with_target[feature] < -0.28:
                selected_features.remove(correlated_feature)
            correlated_features = corr[(corr[feature] < -0.3) | (corr[feature] > 0.3)].index.tolist()
            correlated_features.remove(feature)
            for correlated_feature in correlated_features:
                if correlated_feature in selected_features:
                    selected_features.remove(correlated_feature)
    print(selected_features)
    df = df[[df.columns[0]]+selected_features]
    return df


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    df = pd.read_csv("data1/base_cleaned.csv")
    corr = cal_corr(df)
    plt.figure(figsize=(13, 13))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    # plt.title()
    plt.show()

    print("Original Shape:", df.shape)
    # calculate the training correlation
    df = select_features(df)
    print("Transformed Training Shape:", df.shape)
    # save it to csv
    df.to_csv("base_selected", index=False)



if __name__ == "__main__":
    main()



