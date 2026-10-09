# Kubernetes manifests

```bash
docker build -t greengrid-optimizer:local .
kind load docker-image greengrid-optimizer:local          # or: minikube image load ...
kubectl apply -k k8s
# Optional secrets (chat needs ANTHROPIC_API_KEY, protected endpoints need API_AUTH_TOKEN):
kubectl -n greengrid create secret generic greengrid-secrets --from-literal=API_AUTH_TOKEN="$(openssl rand -hex 16)"
```

The image excludes `data/` and `models/`, so seed the `greengrid-state` PVC with the trained models and processed
data before the API can serve forecasts (the API returns 503 until models exist).

Status: rendered and schema-validated with `kubeconform` (CI job `kubernetes`). **Not applied to a real cluster.**
The API and the updater share one ReadWriteOnce volume, so they are two containers in a single pod (no horizontal
scaling of the API). No Ingress is included.
