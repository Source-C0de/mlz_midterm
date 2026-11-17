#!/usr/bin/env bash
#
# Deploy the Laboratory Quality Prediction API to Google Cloud Run.
#
# Requires: gcloud CLI, authenticated (`gcloud auth login`) and a project with
# billing enabled. Build and deploy happen in the cloud, so nothing needs to
# be installed locally beyond the gcloud CLI.
#
# Usage:
#   GCP_PROJECT_ID=my-project ./deployment/cloudrun-deploy.sh
#
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?set GCP_PROJECT_ID (e.g. GCP_PROJECT_ID=my-project)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${GCP_SERVICE_NAME:-quality-api}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${REPO_ROOT}"

echo "==> Building image ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" .

echo "==> Deploying ${SERVICE} to Cloud Run in ${REGION}"
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --set-env-vars "MODEL_PATH=/app/artifacts/model.joblib"

URL="$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --platform managed \
    --format 'value(status.url)')"

echo
echo "Service URL: ${URL}"
echo

echo "==> GET ${URL}/health"
curl -s "${URL}/health"
echo
echo

echo "==> POST ${URL}/predict"
curl -s -X POST "${URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"type":"L","air_temperature":300.5,"process_temperature":310.2,"rotational_speed":1500,"torque":40.1,"tool_wear":12}'
echo
