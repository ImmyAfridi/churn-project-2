import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(df: pd.DataFrame):
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")
    return model, X_test, y_test

if __name__ == "__main__":
    df = pd.read_csv("data/processed/customers_clean.csv")
    train_model(df)
    print("Model saved.")