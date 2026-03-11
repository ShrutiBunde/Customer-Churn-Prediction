import pandas as pd


def preprocess_data(df):

    df = df.drop(columns=[
        "CustomerID","Count","Country","State","City","Zip Code",
        "Lat Long","Latitude","Longitude",
        "Churn Label","Churn Score","CLTV","Churn Reason"
    ])

    df = df.rename(columns={"Churn Value": "Churn"})

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())
    df = pd.get_dummies(df, drop_first=True)

    return df