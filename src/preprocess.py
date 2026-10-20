import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df["age"] = df["age"].fillna(df["age"].median())
    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
    df = pd.get_dummies(df, columns=["contract"], drop_first=True)
    return df

if __name__ == "__main__":
    raw = pd.read_csv("data/raw/customers.csv")
    clean = clean_data(raw)
    clean.to_csv("data/processed/customers_clean.csv", index=False)
    print(f"Cleaned data: {clean.shape}")