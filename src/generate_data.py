import numpy as np
import pandas as pd

def generate_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "customer_id": range(1, n_rows + 1),
        "age": rng.integers(18, 70, n_rows),
        "tenure_months": rng.integers(1, 72, n_rows),
        "monthly_charges": rng.normal(70, 25, n_rows).round(2),
        "support_calls": rng.poisson(2, n_rows),
        "contract": rng.choice(["Monthly", "Yearly", "Two-Year"], n_rows, p=[0.6, 0.25, 0.15]),
    })

    # Churn probability depends on the features (so the model has something to learn)
    score = (
        0.03 * df["support_calls"]
        + 0.004 * df["monthly_charges"]
        - 0.008 * df["tenure_months"]
        + (df["contract"] == "Monthly") * 0.25
    )
    prob = 1 / (1 + np.exp(-(score - 0.3) * 3))
    df["churn"] = (rng.random(n_rows) < prob).astype(int)

    # Add dirty data on purpose, like real life
    df.loc[rng.choice(n_rows, 60, replace=False), "monthly_charges"] = np.nan
    df.loc[rng.choice(n_rows, 40, replace=False), "age"] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)])  # duplicates

    return df

if __name__ == "__main__":
    data = generate_data()
    data.to_csv("data/raw/customers.csv", index=False)
    print(f"Saved {len(data)} rows to data/raw/customers.csv")