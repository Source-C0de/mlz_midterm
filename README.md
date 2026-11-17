# Laboratory Quality Prediction System

**Predicting Pass / Fail quality-inspection outcomes for manufactured products**

Binary classification on the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 production cycles), served as a containerised REST API.

---

## Table of contents

1. [Problem description](#1-problem-description)
2. [Label provenance: machine failure vs. quality label](#2-label-provenance-machine-failure-vs-quality-label)
3. [Dataset](#3-dataset)
4. [Features](#4-features)
5. [Results](#5-results)
6. [Project structure](#6-project-structure)
7. [Setup](#7-setup)
8. [Running the notebook](#8-running-the-notebook)
9. [Training the model](#9-training-the-model)
10. [Running the API locally](#10-running-the-api-locally)
11. [Containerisation](#11-containerisation)
12. [Kubernetes deployment](#12-kubernetes-deployment)
13. [Cloud deployment](#13-cloud-deployment)
14. [Reproducibility](#14-reproducibility)
15. [Evidence](#15-evidence)
16. [Limitations and future work](#16-limitations-and-future-work)

---

## 1. Problem description

A manufacturing plant runs a production line whose sensors record the operating
conditions of every item produced: ambient and process temperature, rotational
speed, torque and tool wear. At the end of a production cycle the machine may
break down, and a broken machine yields parts that do not meet specification.

**The problem:** decide, *before* the batch ships, whether a product will
**pass** or **fail** quality inspection, using only the process readings
available at the time of production.

**Business value**

- A defective part that reaches the customer costs far more than the small
  cost of one extra inspection.
- Machine breakdowns also destroy tool life and consume schedule, so early
  warning has value beyond the inspection decision itself.
- The plant already logs this data. The missing piece is a model that turns
  those logs into a triage decision.

**How the solution is used.** The output is a **screening tool**, not a
replacement for inspection. Every production run is scored; runs above the
decision threshold are flagged for a detailed quality inspection, the rest
pass through. Because a missed failure ships a defective product while a
false alarm only costs an extra inspection, the operating point is tuned to
favour **recall** (see [§5](#5-results)).

**ML task:** binary classification → **Pass** / **Fail**.

---

## 2. Label provenance: machine failure vs. quality label

> The UCI AI4I 2020 dataset ships one supervised outcome column:
> **`Machine failure`** (1 = the physical machine broke during production,
> 0 = it did not). This is the **actual, ground-truth failure label produced by
> the manufacturer**, and it is the only column used as supervision.
>
> The **quality label** used by this project is **derived from it, not
> independent**: `quality_fail = Machine failure`, so a product is judged
> **`Fail`** exactly when the machine recorded a breakdown during its
> production cycle, and **`Pass`** otherwise. We add **no** subjective or
> heuristic quality judgement — the dataset has no separate human-inspection
> quality column, and we do not fabricate one.
>
> What we *do* add is a set of **engineered process-quality features**
> (out-of-spec tolerance flags and derived ratios, [§4](#4-features)). These
> are **inputs only**: computed from process sensor readings, identically at
> training and inference time, and never part of the label.

Three names are used consistently across the project:

| Name | Meaning | Where it appears |
|---|---|---|
| `Machine failure` | Dataset ground truth (1 = the machine broke) | dataset column, EDA |
| `quality_fail` | Derived supervised target (1 = **Fail**, 0 = **Pass**) | `src/data.py`, training |
| `quality_status` | The API's human-readable verdict (`"Pass"` / `"Fail"`) | `predict.py` responses |

### Why the fault-type columns are excluded

`TWF`, `HDF`, `PWF`, `OSF`, `RNF` record *which* fault occurred. They are set by
the same inspection process that sets `Machine failure` — they are a
**downstream restatement of the outcome**, not an upstream predictor of it.
Using them as model inputs would be reading the answer off the input.

The notebook ([§6](notebook.ipynb)) measures this rather than asserting it. A
random forest trained on the same split, with and without those columns:

| Feature set | # features | PR-AUC | ROC-AUC |
|---|---|---|---|
| Honest (sensors + engineered features) | 14 | **0.8041** | 0.9605 |
| Leaky (+ `TWF/HDF/PWF/OSF/RNF`) | 19 | **0.9790** | 0.9901 |

A **+0.17** PR-AUC jump obtained by reading the answer off the input is not a
model. The subtypes are therefore used **only** for EDA — where they show that
overheating (`HDF`) and over-strain (`OSF`) dominate the failure modes.

---

## 3. Dataset

| Property | Value |
|---|---|
| Name | AI4I 2020 Predictive Maintenance Dataset |
| Source | UCI Machine Learning Repository, dataset id 601 |
| Download URL | <https://archive.ics.uci.edu/static/public/601/data.csv> |
| Licence | CC BY 4.0 |
| Rows | 10,000 production cycles |
| Columns | 14 |
| Missing values | 0 |
| Duplicates | 0 |

### Schema

| Column | Role | Description |
|---|---|---|
| `UID` | ID | Row identifier — dropped |
| `Product ID` | ID | Type prefix + serial — dropped |
| `Type` | Feature | Product tier: `L`, `M`, `H` (60% / 30% / 10%) |
| `Air temperature` | Feature | Ambient temperature (K) |
| `Process temperature` | Feature | Process temperature (K) |
| `Rotational speed` | Feature | Rotational speed (rpm) |
| `Torque` | Feature | Torque (Nm) |
| `Tool wear` | Feature | Accumulated tool wear |
| **`Machine failure`** | **Target (ground truth)** | 1 = the machine broke (339 rows, 3.39%) |
| `TWF`…`RNF` | Fault subtypes | **EDA only** — excluded from the model ([§2](#2-label-provenance-machine-failure-vs-quality-label)) |

### Getting the data

The dataset is **committed to this repository** at `data/raw/ai4i2020.csv`, so
nothing needs to be downloaded to reproduce the results.

To fetch it yourself instead:

```bash
curl -L -o data/raw/ai4i2020.csv \
  "https://archive.ics.uci.edu/static/public/601/data.csv"
```

`train.py` also falls back to downloading the file automatically if it is
missing (`--no-download` disables that and raises an error instead).

> **Note on encoding.** The file published by UCI carries a UTF-8 BOM, which
> makes a plain `pd.read_csv` name the first column `﻿UID` instead of `UID`.
> The BOM has been stripped from the committed copy, so `pd.read_csv` works
> normally. The loader in [`src/data.py`](src/data.py) still reads with
> `encoding="utf-8-sig"`, so it works with either version of the file.

---

## 4. Features

14 columns, defined once in [`src/data.py`](src/data.py) and shared by the
notebook, `train.py` and `predict.py`.

### Raw sensors and derived relationships

| Feature | Definition | Why it helps |
|---|---|---|
| `air_temp` | Air temperature (K) | Ambient conditions |
| `proc_temp` | Process temperature (K) | Core process heat |
| `rot_speed` | Rotational speed (rpm) | Machine dynamics |
| `torque` | Torque (Nm) | Mechanical load |
| `tool_wear` | Tool wear | Tool degradation — a leading indicator |
| `temp_diff` | `proc_temp - air_temp` | Heat actually delivered to the workpiece, not just ambient temperature |
| `torque_per_krpm` | `torque / (rot_speed / 1000)` | Load per unit speed — flags a workpiece too heavy for the rpm |
| `wear_per_load` | `tool_wear / (torque + 1)` | Wear accumulated per unit of work done |
| `air_temp_oos` | `air_temp` outside 296–304 K | Out-of-spec operating flag |
| `proc_temp_oos` | `proc_temp` outside 307–313 K | Out-of-spec operating flag |
| `torque_oos` | `torque` outside 12–62 Nm | Out-of-spec operating flag |
| `rot_speed_oos` | `rot_speed` outside 1300–2200 rpm | Out-of-spec operating flag |
| `type_L`, `type_M` | One-hot of `Type` (`H` = reference) | Product quality tier |

**Dropped:** `UID`, `Product ID`, and the five fault-subtype columns.

### Why the out-of-spec thresholds leak nothing

The thresholds are taken from the **univariate feature distributions** — the
bulk of the operating envelope — and were chosen **without reference to the
target**. The resulting flag rates are a meaningful minority of rows, which
confirms they describe something real rather than being fitted to the label:

| Flag | Bound | Observed range | Flag rate |
|---|---|---|---|
| `air_temp_oos` | 296–304 K | 295.3 – 304.5 | 2.18% |
| `proc_temp_oos` | 307–313 K | 305.7 – 313.8 | 3.21% |
| `torque_oos` | 12–62 Nm | 3.0 – 63.0 | 1.47% |
| `rot_speed_oos` | 1300–2200 rpm | 1186 – 2886 | 3.07% |

---

## 5. Results

### Model comparison

Cross-validated on the **training split** (5-fold stratified CV, metric
PR-AUC). The winner is chosen here, not by test performance.

| Model | Family | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Recall | Precision | F2 |
|---|---|---|---|---|---|---|---|
| **HistGradientBoosting** | boosting | **0.8445** | **0.8966** | 0.9850 | 0.8971 | 0.6162 | 0.8221 |
| RandomForest | tree ensemble | 0.8255 | 0.8525 | 0.9808 | 0.8382 | 0.7500 | 0.8190 |
| LogisticRegression | linear | 0.4760 | 0.4777 | 0.9392 | 0.7206 | 0.3603 | 0.6005 |
| *Always "Pass" (baseline)* | — | *0.034* | *0.034* | *0.500* | *0.000* | *0.000* | *0.000* |

> **Accuracy is not a meaningful metric for this task.** At 3.39% prevalence a
> model that always answers "Pass" scores 96.6% accuracy while catching no
> failures at all. PR-AUC is used throughout instead.

### Final model

**HistGradientBoostingClassifier**, threshold tuned to **0.41** on
out-of-fold training predictions.

| Metric | Value |
|---|---|
| PR-AUC | **0.8966** |
| ROC-AUC | 0.9850 |
| Recall | 0.8971 |
| Precision | 0.6162 |
| F2 | 0.8221 |
| Accuracy | 0.9775 *(misleading — see above)* |

Confusion matrix on the held-out test set:

| | Predicted Pass | Predicted Fail |
|---|---|---|
| **Actual Pass** | 1,894 (TN) | 38 (FP) |
| **Actual Fail** | 7 (FN) | 61 (TP) |

61 of 68 real failures were caught. 38 good products were flagged for extra
inspection — the price of that recall, and a price worth paying when a
missed failure ships a defective product.

### Top features by permutation importance

| Feature | Importance |
|---|---|
| `temp_diff` | 0.5014 |
| `rot_speed` | 0.4807 |
| `torque` | 0.2506 |
| `tool_wear` | 0.2161 |
| `torque_per_krpm` | 0.1972 |

The failure signal lives in the *relationship* between readings, not in any
single sensor: a large process-minus-air temperature gap and a high
rotational speed are the dominant signals. The tuned logistic regression
agrees on direction, with positive coefficients on `temp_diff` and
`torque_per_krpm`.

### Why F2, and why a tuned threshold

The model is a screening tool, so the operating point is chosen for recall.
`class_weight="balanced"` already handles the class imbalance during fitting,
but it shifts the probability scale, so the 0.5 default is not the right
cut-off. The threshold is swept over **out-of-fold** predictions and the F2
maximum is selected — F2 weights recall four times as heavily as precision.

**No SMOTE / `imblearn`.** With only 339 positive rows, synthetic oversampling
can generate near-duplicate positives that then land in the validation fold,
producing optimistically biased CV scores. `class_weight` addresses the
imbalance without fabricating data.

---

## 6. Project structure

```
.
├── README.md                  # this file
├── notebook.ipynb             # EDA, feature importance, model selection + tuning
├── train.py                   # CLI trainer -> artifacts/model.joblib
├── predict.py                 # Flask REST API
├── build_notebook.py          # regenerates notebook.ipynb
├── requirements.txt           # runtime dependencies
├── requirements-dev.txt       # extra packages for the notebook
├── Dockerfile                 # container image for the API
├── .dockerignore
├── .gitignore
├── render.yaml                # Render blueprint (public URL in ~3 min)
├── src/
│   ├── data.py                # label rule + shared feature matrix
│   └── modeling.py            # pipelines, grids, metrics
├── data/
│   └── raw/ai4i2020.csv       # committed dataset
├── artifacts/                 # model.joblib + metrics.json (committed)
├── deployment/
│   ├── k8s/                   # Deployment, Service, Ingress, kustomization, deploy-k3d.sh
│   ├── docker-compose.yaml
│   ├── cloudrun-deploy.sh     # Google Cloud Run one-liner
│   ├── capture_evidence.py    # regenerates evidence/predict-curl.txt
│   ├── make_screenshot.py     # regenerates evidence/screenshot.png
│   └── make_k8s_screenshot.py # regenerates evidence/k8s-screenshot.png
└── evidence/                  # proof artefacts for the deployed service
```

`src/` exists for one reason: `train.py` and `predict.py` must produce and
consume an **identical** feature matrix. One shared `build_features()`
prevents silent train/serve skew — the notebook asserts they still match.

---

## 7. Setup

Requires Python 3.11+ (developed on 3.13).

### Create a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd)**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt          # runtime: training + API
pip install -r requirements-dev.txt      # adds matplotlib, seaborn, jupyterlab
```

`requirements-dev.txt` includes `requirements.txt`, so the second command is
sufficient on its own.

---

## 8. Running the notebook

```bash
jupyter lab notebook.ipynb
```

Run all cells top to bottom. The notebook is deterministic (`SEED = 42`) and
covers:

1. Problem statement and business context
2. **Label provenance** — machine failure vs. the derived quality label
3. Dataset overview
4. **EDA** — schema, missing values, duplicates, ranges, target imbalance,
   fault-subtype breakdown, distributions split by target, correlations,
   initial feature importance
5. **Feature engineering** — the 14-feature matrix and out-of-spec flag rates
6. **Leakage check** — the honest-vs-leaky PR-AUC comparison
7. Stratified train/test split
8. **Model selection** — 3 candidates × `GridSearchCV`, threshold sweep,
   final metrics, confusion matrix, PR and ROC curves
9. **Feature importance** — permutation importance + logistic coefficients
10. Results summary, limitations, reproducibility, consistency guard

The notebook writes the same `artifacts/model.joblib` that `train.py` does.

---

## 9. Training the model

```bash
python train.py
```

Full grid search over all three candidates. Output:

```
=== Model comparison (5-fold CV on train, metric: PR-AUC) ===
model        cv_pr_auc   roc_auc      f2   recall
-------------------------------------------------
hgb             0.8445    0.9850  0.8221   0.8971
rf              0.8255    0.9808  0.8190   0.8382
logreg          0.4760    0.9392  0.6005   0.7206

=== Final model: hgb  (threshold=0.41) ===
Test PR-AUC  : 0.8966
...
Saved model     -> artifacts/model.joblib
Saved metrics   -> artifacts/metrics.json
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--fast` | Skip the grid search (smoke test, ~30 s) |
| `--data PATH` | Use a different raw CSV |
| `--out-dir PATH` | Write artifacts elsewhere |
| `--cv-folds N` | Change the number of CV folds (default 5) |
| `--seed N` | Change the random seed (default 42) |
| `--no-download` | Fail instead of downloading a missing dataset |

---

## 10. Running the API locally

```bash
python predict.py                       # dev server on :8000
```

```bash
gunicorn --bind 0.0.0.0:8000 --workers 2 --threads 4 predict:app   # production
```

> gunicorn runs on Linux/macOS and inside the Docker image. On Windows use
> `python predict.py` for local development.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness / readiness probe (used by Docker and Kubernetes) |
| `GET` | `/` | API index with an example request |
| `POST` | `/predict` | Predict one product |
| `POST` | `/predict/batch` | Predict a list (max 500 items) |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature":300.5,"process_temperature":310.2,
       "rotational_speed":1500,"torque":40.1,"tool_wear":12}'
```

```json
{
  "quality_status": "Pass",
  "quality_label": 0,
  "machine_failure_probability": 0.004675,
  "pass_probability": 0.995325,
  "threshold": 0.41,
  "explanation": {
    "air_temp_oos": 0,
    "proc_temp_oos": 0,
    "torque_oos": 0,
    "rot_speed_oos": 0
  }
}
```

**Field semantics**

- `quality_label` — `1` = **Fail**, `0` = **Pass** (matches `quality_fail`).
- `quality_status` — the same verdict as a string.
- `machine_failure_probability` — P(the machine broke). The model predicts
  `quality_fail`, and `quality_fail = 1` exactly when `Machine failure = 1`,
  so this probability is identical either way. The name keeps the ground-truth
  provenance visible in the API instead of hiding it behind a vague "risk score".
- `explanation` — which out-of-spec flags the run triggered.

A request with a missing field, a non-numeric value or a non-finite number
returns `400` with a message naming the offending field:

```json
{"error": "bad_request", "message": "Field 'air_temperature' must be a number, got 'hot'"}
```

---

## 11. Containerisation

The image trains the model **at build time**, so it ships a ready-to-serve
artifact and needs no volume mount. Training is deterministic (fixed seed), so
rebuilds produce the same model.

```bash
# Build
docker build -t quality-api:1.0 .

# Run (published on :8080)
docker run -d --name quality-api -p 8080:8000 quality-api:1.0

# Check it is healthy
docker ps                       # STATUS shows "(healthy)"
curl -s http://localhost:8080/health

# Predict
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature":300.5,"process_temperature":310.2,
       "rotational_speed":1500,"torque":40.1,"tool_wear":12}'

# Stop and clean up
docker rm -f quality-api
```

The image runs **gunicorn** (2 workers × 4 threads) rather than the Flask
development server, and declares a `HEALTHCHECK` against `/health`.

To use a pre-trained model instead of building one, mount it and point
`MODEL_PATH` at it:

```bash
docker run -d -p 8080:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  -e MODEL_PATH=/app/artifacts/model.joblib \
  quality-api:1.0
```

Docker Compose is also provided:

```bash
docker compose -f deployment/docker-compose.yaml up --build
```

---

## 12. Kubernetes deployment

Manifests live in [`deployment/k8s/`](deployment/k8s):

| File | Contents |
|---|---|
| `deployment.yaml` | Deployment, 2 replicas, readiness + liveness probes on `/health`, resource requests/limits |
| `service.yaml` | ClusterIP service on port 80 → 8000 |
| `ingress-nginx.yaml` | nginx Ingress for `quality.local` |
| `kustomization.yaml` | Ties them together |
| `deploy-k3d.sh` | One-command deploy to a local containerised k3s cluster (no cloud account needed) |

### Verified on a live Kubernetes cluster

The manifests in this repository were applied to a **real Kubernetes cluster**
(a single-node k3s cluster created with k3d) and served live traffic. The
transcript of that run is committed at
[`evidence/k8s-cluster.txt`](evidence/k8s-cluster.txt), and the screenshot at
[`evidence/k8s-screenshot.png`](evidence/k8s-screenshot.png) shows the cluster
state together with the JSON the service returned:

```
$ kubectl get pods -o wide
NAME                           READY   STATUS    RESTARTS   AGE   IP           NODE
quality-api-546745549f-gmx99   1/1     Running   0          3m    10.42.0.11   k3d-quality-server-0
quality-api-546745549f-x5z5g   1/1     Running   0          3m    10.42.0.12   k3d-quality-server-0

$ curl -s -H 'Host: quality.local' http://127.0.0.1:8081/health
{"status":"ok","model":"hgb","threshold":0.41, ...}

$ curl -s -H 'Host: quality.local' -X POST http://127.0.0.1:8081/predict \
    -H 'Content-Type: application/json' \
    -d '{"type":"L","air_temperature":298.9,"process_temperature":309.1,
         "rotational_speed":2861,"torque":4.6,"tool_wear":143}'
{"quality_status":"Fail","quality_label":1,"machine_failure_probability":0.992219,
 "explanation":{"proc_temp_oos":1,"rot_speed_oos":1,"torque_oos":1}}
```

The request path is the full production one: **Ingress controller → Service →
Deployment → gunicorn → Flask**, with both replicas passing their probes.

### Deploy with one command (k3d)

`deployment/k8s/deploy-k3d.sh` performs the whole sequence — build the image,
create the cluster, import the image into it, apply the manifests and wait for
the rollout:

```bash
bash deployment/k8s/deploy-k3d.sh
```

Requires only Docker Desktop and `kubectl`; no cloud account, no credentials.

### With kind

```bash
kind create cluster --name quality
docker build -t quality-api:1.0 .
kind load docker-image quality-api:1.0 --name quality
kubectl apply -k deployment/k8s/
kubectl port-forward svc/quality-api 8080:80
curl -s http://localhost:8080/health
```

### With minikube

```bash
minikube start
eval $(minikube docker-env)
docker build -t quality-api:1.0 .
kubectl apply -k deployment/k8s/
kubectl port-forward svc/quality-api 8080:80
```

### On a remote cluster

Push the image to a registry and update the `image:` field in
`deployment/k8s/deployment.yaml`:

```bash
docker tag quality-api:1.0 gcr.io/<project>/quality-api:1.0
docker push gcr.io/<project>/quality-api:1.0
```

---

## 13. Cloud deployment

The service is containerised and ready for any container platform. The
`Dockerfile` in the repository root is the single artefact every path below
uses.

> **On the public URL.** A live public endpoint is bound to a cloud account, so
> none can be published from this repository without your credentials — and
> none is fabricated. Everything needed to produce one in a few minutes is
> here: the code, the container image definition, and the deployment manifests.
> [§15](#15-evidence) shows the service running and answering requests inside
> the container, and [§12](#12-kubernetes-deployment) shows it running inside a
> **live Kubernetes cluster** behind an ingress controller — the same
> deployable artefact the cloud hosts.

### Google Cloud Run

Requires the `gcloud` CLI, `gcloud auth login`, and a project with billing
enabled. The build and deploy both happen in the cloud.

```bash
GCP_PROJECT_ID=my-project ./deployment/cloudrun-deploy.sh
```

The script builds the image, deploys the service with `--allow-unauthenticated`,
prints the service URL, and curls `/health` and `/predict` to verify it.

### Render (fastest route to a public URL)

[`render.yaml`](render.yaml) is a Render blueprint at the repository root, and
the repository is already published at <https://github.com/Source-C0de/mlz_midterm>.
To get a public URL:

1. Sign in at <https://render.com> (a free account is enough).
2. **New → Blueprint**, then point it at `Source-C0de/mlz_midterm`.
3. Apply. Render builds the `Dockerfile` and serves the API at a public
   `https://quality-api.onrender.com` URL.

Once it is live, verify it with:

```bash
curl -s https://quality-api.onrender.com/health
curl -s -X POST https://quality-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature":300.5,"process_temperature":310.2,
       "rotational_speed":1500,"torque":40.1,"tool_wear":12}'
```

> Render's free tier sleeps the service after inactivity, so the first request
> can take ~30 s to wake it.

### Manual

Any platform that runs a container works: AWS ECS/Fargate, Azure Container
Apps, Google Cloud Run, Fly.io, Railway. The only requirement is that the
container listens on `$PORT` (default `8000`) and exposes `/health`.

---

## 14. Reproducibility

- `SEED = 42` is fixed everywhere — the split, the CV folds, the models and
  the permutation importance.
- Package versions are pinned in `requirements.txt`.
- The dataset is **committed** at `data/raw/ai4i2020.csv`; nothing needs to be
  downloaded, though `train.py` can fetch it automatically.
- The **trained model is committed** too (`artifacts/model.joblib`, 104 KB), so
  a fresh clone can serve requests immediately without retraining.
- `train.py` reproduces `artifacts/model.joblib` from scratch, and the
  notebook's training cells mirror it line for line.
- The notebook ends with a consistency guard asserting that the feature list
  still matches `src/data.py` and that no fault-subtype column has leaked in.

Verify from a clean checkout:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# The committed model is ready to serve straight away:
python predict.py            # in another terminal
curl -s http://localhost:8000/health

# Or retrain from scratch and confirm the metrics reproduce:
python train.py
```

Run all commands **from the repository root** — `train.py` and `predict.py`
resolve `data/raw/` and `artifacts/` relative to the working directory.

---

## 15. Evidence

| File | What it shows |
|---|---|
| [`evidence/train-output.txt`](evidence/train-output.txt) | Full training run — model comparison table and final metrics |
| [`evidence/predict-curl.txt`](evidence/predict-curl.txt) | Live API responses: `/health`, `/predict` on a normal product, `/predict` on a real failure row, `/predict/batch`, and both `400` error paths |
| [`evidence/docker-build.log`](evidence/docker-build.log) | Docker image build, including the model training that happens inside it |
| [`evidence/docker-run-curl.txt`](evidence/docker-run-curl.txt) | The running container answering requests |
| [`evidence/k8s-dry-run.txt`](evidence/k8s-dry-run.txt) | Kubernetes manifest validation |
| [`evidence/k8s-cluster.txt`](evidence/k8s-cluster.txt) | **Live Kubernetes deployment** — cluster, pods, service and ingress state, plus the HTTP responses served through the cluster's ingress controller |
| [`evidence/k8s-screenshot.png`](evidence/k8s-screenshot.png) | The service running inside a Kubernetes cluster: `kubectl get pods/svc/ingress` and the JSON it returned over the ingress |
| [`evidence/screenshot.png`](evidence/screenshot.png) | The deployed service: healthy container, `/health`, and both predictions |

Regenerate the API transcript against a running service:

```bash
python deployment/capture_evidence.py --base-url http://127.0.0.1:8000
```

Regenerate the screenshot:

```bash
python deployment/make_screenshot.py --base-url http://127.0.0.1:8080
```

---

## 16. Limitations and future work

- **Only 339 failures.** The metric estimates carry real variance. Stratified
  5-fold CV and a fixed seed keep the comparison reproducible, but a single
  PR-AUC figure should not be over-read.
- **The label is the breakdown label.** AI4I has no independent
  human-inspection quality column, so `quality_fail` is derived from
  `Machine failure` ([§2](#2-label-provenance-machine-failure-vs-quality-label)).
  A plant with real inspection verdicts would retrain on those instead — the
  pipeline is unchanged.
- **The fault subtypes are unavailable at prediction time** in the real world,
  which is exactly why they are excluded.
- **No time dimension.** Rows are independent production cycles; the model
  does not model drift or the tool-wear trajectory. A sequential model, or a
  rolling tool-wear feature, would be the natural next step.
- **Threshold drift.** The threshold was tuned on this dataset's operating
  envelope and should be re-tuned as the plant changes.
- **Real failure rates are much higher than 3.39%.** On a line with a 20%
  failure rate the same classifier would need different thresholds and would
  look considerably better than these numbers.
- **Precision is the cost of recall.** At 0.62 precision the model flags
  roughly 1 in 6 good products for inspection. Raising the threshold trades
  recall for a lower false-alarm rate; the trade-off should be set with the
  quality department, not by a default.

---

## References

- S. Matzka, *"Explainable Artificial Intelligence for Predictive Maintenance
  Applications"*, International Conference on Artificial Intelligence for
  Industries, 2020. DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)
- UCI Machine Learning Repository — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (CC BY 4.0)
