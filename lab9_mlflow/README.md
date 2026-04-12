# Lab 9 - MLflow on Kind

This folder contains the reproducible assets for Lab 9.

## Files
- `deploy_lab9.sh` - creates the Kind cluster and installs PostgreSQL, MinIO, and MLflow with Helm
- `train_with_mlflow.py` - trains the wine-quality model and logs params, metrics, and artifacts to MLflow
- `requirements.txt` - Python dependencies for the MLflow logging script

## Usage
1. Export the required environment variables:
   - `PG_PASSWORD`
   - `MINIO_ROOT_USER`
   - `MINIO_ROOT_PASSWORD`
2. Run `bash lab9_mlflow/deploy_lab9.sh`
3. Port-forward MLflow and MinIO:
   - `kubectl -n mlflow port-forward svc/mlflow 8080:80`
   - `kubectl -n mlflow port-forward svc/mlflow-minio 9000:9000 9001:9001`
4. Run `python lab9_mlflow/train_with_mlflow.py`
