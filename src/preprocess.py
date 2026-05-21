import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():
    df = pd.read_csv("data/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )