from src.pipeline import main

import pandas as pd
from src.generate_data import generate_data
from src.preprocess import clean_data
from src.train import train_model
from src.evaluate import evaluate_model

def run():
    print("1. Generating data...")
    raw = generate_data()
    raw.to_csv("data/raw/customers.csv", index=False)

    print("2. Cleaning data...")
    clean = clean_data(raw)
    clean.to_csv("data/processed/customers_clean.csv", index=False)

    print("3. Training model...")
    model, X_test, y_test = train_model(clean)

    print("4. Evaluating...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run()