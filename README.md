# Customer Churn Prediction

End-to-end ML pipeline on synthetic customer data.

## What it does
1. Generates raw customer data (with missing values and duplicates)
2. Cleans and encodes the data
3. Trains a Random Forest classifier
4. Evaluates with a classification report and ROC-AUC

## How to run
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py

## Project structure
- src/ : data generation, preprocessing, training, evaluation
- data/ : raw and processed CSVs
- models/ : saved models
- main.py : runs the full pipeline