#!/usr/bin/env bash
set -euo pipefail

ROLL_NO="${ROLL_NO:-2022BCS0016}"
CLUSTER_NAME="${CLUSTER_NAME:-mlops-${ROLL_NO,,}}"
NAMESPACE="${NAMESPACE:-mlflow}"
PG_USER="${PG_USER:-mlflow}"
PG_DATABASE="${PG_DATABASE:-mlflow}"
MINIO_BUCKET="${MINIO_BUCKET:-mlflow-${ROLL_NO,,}}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:?Set MINIO_ROOT_USER before running this script}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:?Set MINIO_ROOT_PASSWORD before running this script}"
PG_PASSWORD="${PG_PASSWORD:?Set PG_PASSWORD before running this script}"

if ! kind get clusters | grep -qx "$CLUSTER_NAME"; then
  kind create cluster --name "$CLUSTER_NAME"
fi

helm repo add community-charts https://community-charts.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add bitnami https://charts.bitnami.com/bitnami >/dev/null 2>&1 || true
helm repo update >/dev/null

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install mlflow-postgresql bitnami/postgresql \
  --namespace "$NAMESPACE" \
  --set auth.username="$PG_USER" \
  --set auth.password="$PG_PASSWORD" \
  --set auth.postgresPassword="$PG_PASSWORD" \
  --set auth.database="$PG_DATABASE" \
  --set primary.persistence.enabled=false

helm upgrade --install mlflow-minio bitnami/minio \
  --namespace "$NAMESPACE" \
  --set auth.rootUser="$MINIO_ROOT_USER" \
  --set auth.rootPassword="$MINIO_ROOT_PASSWORD" \
  --set defaultBuckets="$MINIO_BUCKET" \
  --set persistence.enabled=false

helm upgrade --install mlflow community-charts/mlflow \
  --namespace "$NAMESPACE" \
  --set backendStore.databaseMigration=true \
  --set backendStore.databaseConnectionCheck=true \
  --set backendStore.postgres.enabled=true \
  --set backendStore.postgres.host=mlflow-postgresql \
  --set backendStore.postgres.port=5432 \
  --set backendStore.postgres.database="$PG_DATABASE" \
  --set backendStore.postgres.user="$PG_USER" \
  --set backendStore.postgres.password="$PG_PASSWORD" \
  --set backendStore.postgres.driver=psycopg2 \
  --set artifactRoot.s3.enabled=true \
  --set artifactRoot.s3.bucket="$MINIO_BUCKET" \
  --set artifactRoot.s3.awsAccessKeyId="$MINIO_ROOT_USER" \
  --set artifactRoot.s3.awsSecretAccessKey="$MINIO_ROOT_PASSWORD" \
  --set-string extraEnvVars.AWS_DEFAULT_REGION=us-east-1 \
  --set-string extraEnvVars.AWS_EC2_METADATA_DISABLED=true \
  --set-string extraEnvVars.MLFLOW_S3_ENDPOINT_URL=http://mlflow-minio:9000 \
  --set postgresql.enabled=false

kubectl wait --namespace "$NAMESPACE" --for=condition=Ready pod -l app.kubernetes.io/instance=mlflow-postgresql --timeout=300s
kubectl wait --namespace "$NAMESPACE" --for=condition=Ready pod -l app.kubernetes.io/instance=mlflow-minio --timeout=300s
kubectl wait --namespace "$NAMESPACE" --for=condition=Ready pod -l app.kubernetes.io/instance=mlflow --timeout=300s

echo "Cluster ready: $CLUSTER_NAME"
echo "Namespace: $NAMESPACE"
echo "MLflow service: kubectl -n $NAMESPACE port-forward svc/mlflow 8080:80"
echo "MinIO console: kubectl -n $NAMESPACE port-forward svc/mlflow-minio 9000:9000 9001:9001"
