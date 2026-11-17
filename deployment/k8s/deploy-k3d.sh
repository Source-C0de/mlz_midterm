#!/usr/bin/env bash
# Deploy the quality API to a local Kubernetes cluster using k3d (containerised k3s).
#
# This is the credential-free way to exercise the full Kubernetes deployment path
# (Deployment -> Service -> Ingress -> 2 running replicas -> live HTTP requests).
#
# Prerequisites: Docker Desktop running, kubectl on PATH.
#
# Usage:
#   bash deployment/k8s/deploy-k3d.sh
#   bash deployment/k8s/deploy-k3d.sh --cluster quality --port 8081
#
# Then:
#   curl -H 'Host: quality.local' http://127.0.0.1:8081/health
#   curl -H 'Host: quality.local' -X POST http://127.0.0.1:8081/predict \
#        -H 'Content-Type: application/json' \
#        -d '{"type":"L","air_temperature":300.5,"process_temperature":310.2,
#             "rotational_speed":1500,"torque":40.1,"tool_wear":12}'

set -euo pipefail

CLUSTER="quality"
PORT="8081"
IMAGE="quality-api:1.0"
HOST="quality.local"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2 ;;
    --port)    PORT="$2";    shift 2 ;;
    --image)   IMAGE="$2";   shift 2 ;;
    --host)    HOST="$2";    shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# k3d's own images are referenced by a build-specific tag; the published
# "latest" tags are tagged locally under that name so the pull resolves offline.
K3D_VERSION="949e9c6c"

echo "==> Pulling k3d + k3s images"
docker pull rancher/k3d:latest
docker pull rancher/k3s:v1.30.2-k3s1
docker pull rancher/k3d-proxy:latest
docker pull rancher/k3d-tools:latest
docker tag rancher/k3d-proxy:latest "rancher/k3d-proxy:${K3D_VERSION}"
docker tag rancher/k3d-tools:latest "rancher/k3d-tools:${K3D_VERSION}"

echo "==> Building the application image"
docker build -t "$IMAGE" .

echo "==> Creating the k3d cluster '${CLUSTER}'"
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint k3d \
  -v /var/run/docker.sock:/var/run/docker.sock rancher/k3d:latest \
  cluster create "$CLUSTER" --port "${PORT}:80@servers:*:proxy" --timeout 5m

echo "==> Exporting the kubeconfig"
mkdir -p "${HOME}/.kube"
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint k3d \
  -v /var/run/docker.sock:/var/run/docker.sock rancher/k3d:latest \
  kubeconfig get "$CLUSTER" > "${HOME}/.kube/config"

# k3d advertises the API on the load balancer's published host port, which is
# what the host machine can actually reach (the container IP is not routable
# from Windows/macOS hosts).
API_PORT="$(docker port "${CLUSTER}-serverlb" 6443/tcp | head -n1 | sed 's/.*://')"
python - "$API_PORT" <<'PY'
import pathlib, re, sys
port = sys.argv[1]
path = pathlib.Path.home() / ".kube" / "config"
text = path.read_text(encoding="utf-8")
text = re.sub(r"(?m)^(\s*)server: https://.*$", rf"\1server: https://127.0.0.1:{port}", text)
path.write_text(text, encoding="utf-8")
print(f"kubeconfig -> https://127.0.0.1:{port}")
PY

echo "==> Importing the application image into the cluster"
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint k3d \
  -v /var/run/docker.sock:/var/run/docker.sock rancher/k3d:latest \
  image import "$IMAGE" -c "$CLUSTER"

echo "==> Applying the manifests"
kubectl apply \
  -f deployment/k8s/deployment.yaml \
  -f deployment/k8s/service.yaml \
  -f deployment/k8s/ingress-nginx.yaml

echo "==> Waiting for the rollout"
kubectl rollout status deployment/quality-api --timeout=180s
kubectl get pods -o wide

echo
echo "The service is reachable through the cluster's Ingress controller:"
echo "  curl -H 'Host: ${HOST}' http://127.0.0.1:${PORT}/health"
echo "  curl -H 'Host: ${HOST}' -X POST http://127.0.0.1:${PORT}/predict \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"type\":\"L\",\"air_temperature\":300.5,\"process_temperature\":310.2,\"rotational_speed\":1500,\"torque\":40.1,\"tool_wear\":12}'"
