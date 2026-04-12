import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROLL_NO = "2022BCS0016"
STUDENT_NAME = "Srinivas Raghav V C"
MODEL_TYPE = "Lasso"
TEST_SIZE = 0.2
RANDOM_STATE = 42
ALPHA = 0.1
APPLY_SCALING = True
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", f"{ROLL_NO}-wine-quality-lab9")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    artifacts_dir = Path("lab9_mlflow") / "output"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/winequality-red.csv", sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    if APPLY_SCALING:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    model = Lasso(alpha=ALPHA)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "roll_no": ROLL_NO,
        "student_name": STUDENT_NAME,
        "model_type": MODEL_TYPE,
        "mse": round(mse, 6),
        "r2_score": round(r2, 6),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    metrics_path = artifacts_dir / "metrics.json"
    model_path = artifacts_dir / "model.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with mlflow.start_run(run_name=f"{ROLL_NO}-wine-quality"):
        mlflow.set_tag("student_name", STUDENT_NAME)
        mlflow.set_tag("roll_no", ROLL_NO)
        mlflow.log_params(
            {
                "model_type": MODEL_TYPE,
                "alpha": ALPHA,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "apply_scaling": APPLY_SCALING,
            }
        )
        mlflow.log_metrics({"mse": mse, "r2_score": r2})
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.sklearn.log_model(model, artifact_path="wine_quality_model")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
