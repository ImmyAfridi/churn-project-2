from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_TARGET = "Churn"
DEFAULT_MODEL_PATH = Path("models") / "churn_model.joblib"
DEFAULT_METRICS_PATH = Path("models") / "churn_metrics.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def generate_synthetic_churn_data(
    n_samples: int = 2000,
    random_state: int = 42,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Create a realistic synthetic churn dataset when no CSV is present."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.72, 0.28],
        flip_y=0.05,
        random_state=random_state,
    )

    feature_names = [
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "internet_service",
        "payment_method",
        "support_calls",
        "avg_usage_hours",
        "is_premium_customer",
        "last_login_days",
        "satisfaction_score",
        "total_refunds",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["Churn"] = y
    df["contract_type"] = pd.cut(df["contract_type"], bins=3, labels=["Month-to-month", "One year", "Two year"])
    df["internet_service"] = pd.cut(df["internet_service"], bins=3, labels=["DSL", "Fiber", "None"])
    df["payment_method"] = pd.cut(df["payment_method"], bins=3, labels=["Bank transfer", "Credit card", "Electronic check"])
    df["is_premium_customer"] = (df["is_premium_customer"] > 0.5).astype(int)
    df["total_refunds"] = df["total_refunds"].round(2)

    # Make the target more interpretable: 1 means churn, 0 means retained.
    df["Churn"] = df["Churn"].astype(int)
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)
    return df


def _candidate_data_paths(raw_path: str | Path | None = None) -> list[Path]:
    root = _project_root()
    paths: list[Path] = []
    if raw_path is not None:
        paths.append(Path(raw_path))
    paths.extend(
        [
            root / "data" / "raw",
            root / "data" / "processed",
            root / "data",
            root,
        ]
    )
    seen: set[Path] = set()
    result: list[Path] = []
    for location in paths:
        if not location.exists():
            continue
        if location.is_file():
            resolved = location.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(location)
        else:
            for file in sorted(location.rglob("*.csv")):
                resolved = file.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    result.append(file)
    return result


def load_churn_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load a churn dataset from CSV or synthesize a dataset if no file is present."""
    candidates = _candidate_data_paths(data_path)
    if data_path is not None:
        explicit = Path(data_path)
        if explicit.exists():
            return pd.read_csv(explicit)

    for path in candidates:
        try:
            frame = pd.read_csv(path)
            if DEFAULT_TARGET in frame.columns:
                return frame
        except Exception:
            continue

    synthetic_path = _project_root() / "data" / "raw" / "synthetic_churn.csv"
    return generate_synthetic_churn_data(path=synthetic_path)


def prepare_features(df: pd.DataFrame, target_col: str = DEFAULT_TARGET) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate target and features, with a default target name fallback."""
    frame = df.copy()
    if target_col not in frame.columns:
        if "target" in frame.columns:
            target_col = "target"
        elif "churn" in frame.columns:
            target_col = "churn"
        else:
            raise ValueError(f"Target column '{target_col}' was not found in dataframe columns: {list(frame.columns)}")
    feature_frame = frame.drop(columns=[target_col])
    target = frame[target_col]
    feature_names = [col for col in feature_frame.columns if col != "customer_id"]
    return feature_frame[feature_names], target, feature_names


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_churn_model(
    data: pd.DataFrame | None = None,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train a churn model and return the fitted estimator and summary metrics."""
    frame = load_churn_data() if data is None else data
    X, y, _ = prepare_features(frame, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "test_size": len(X_test),
        "train_size": len(X_train),
    }
    return model, metrics


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def save_model(model: Pipeline, path: str | Path | None = None) -> Path:
    destination = Path(path) if path is not None else DEFAULT_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def save_metrics(metrics: dict[str, Any], path: str | Path | None = None) -> Path:
    destination = Path(path) if path is not None else DEFAULT_METRICS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def run_training_pipeline(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    data = load_churn_data(data_path)
    model, metrics = train_churn_model(data)
    saved_model = save_model(model, model_path)
    saved_metrics = save_metrics(metrics, metrics_path)
    return {
        "model_path": str(saved_model),
        "metrics_path": str(saved_metrics),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train and evaluate a customer churn model.")
    parser.add_argument("--data", type=str, default=None, help="Optional path to a CSV churn dataset.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Where to save the trained model.")
    parser.add_argument("--metrics-path", type=str, default=str(DEFAULT_METRICS_PATH), help="Where to save the training metrics.")
    args = parser.parse_args(argv)

    result = run_training_pipeline(data_path=args.data, model_path=args.model_path, metrics_path=args.metrics_path)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
