# Update an existing deployment to serve HDF5

Prediction now loads `models/versions/<active-version>/model.h5`. It follows the
registry during promotion and rollback; it does not use the fixed top-level
`models/brain_mri_classifier.h5` submission copy. Training and evaluation retain
their native Keras default, and training continues exporting both formats.

Verified on Azure on 2026-11-26: active model `v002`, one API replica, healthy public
HTTPS endpoint reporting `model_artifact: "model.h5"`. All 58 targeted local tests
passed. Four public sample predictions matched the previous Keras classes and
confidence scores exactly; two samples were misclassified in both formats, so this
is a compatibility check, not evidence of improved accuracy. See the
[verification report](../../reports/deployment/hdf5_20261126.json).

This update used Azure's VM agent because SSH was timing out. It backed up the
three original source files at
`/home/azureuser/brain-mri-h5-backup-20261126T133929Z.tar.gz` and retained the previous
image as `brain-mri-api:h5-backup-20261126T133929Z`. The repeatable SSH workflow and
rollback procedure are documented below. No model weights or training data were
replaced, and the VM size, schedule and API replica count were unchanged.

The runtime changes are limited to:

- `src/prediction.py` — explicitly load HDF5 and record the loaded filename.
- `api/schemas.py` — expose the filename in the health response.
- `api/routers/system.py` — report the loaded predictor's artifact.

## Prepare the update on your laptop

From the project root, after the prediction/registry/API tests pass:

```powershell
tar -czf brain-mri-h5-update.tar.gz src/prediction.py api/schemas.py api/routers/system.py
scp -4 -o ConnectTimeout=10 brain-mri-h5-update.tar.gz azureuser@YOUR-SITE-HOST:~/
ssh -4 -o ConnectTimeout=10 azureuser@YOUR-SITE-HOST
```

This archive contains only those three source files. It contains no environment
file, model, dataset, database or Docker volume. Keep the existing model artifacts
on the VM. Before applying it, ensure these files have no unrelated server-only
edits that would be overwritten.

## Back up and update on the VM

Run these commands on the VM in one shell. They retain the current API replica
count and save both the original source and running API image for rollback.

```bash
cd ~/brain-mri
dc() { docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml "$@"; }
replicas=$(dc ps -q api | wc -l)
test "$replicas" -gt 0 || { echo "No running API replicas; inspect the stack first."; exit 1; }
stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup="$HOME/brain-mri-h5-backup-$stamp.tar.gz"
old_image=$(docker inspect --format '{{.Image}}' "$(dc ps -q api | head -n 1)")
rollback_tag="brain-mri-api:h5-backup-$stamp"
docker image tag "$old_image" "$rollback_tag"
tar -czf "$backup" src/prediction.py api/schemas.py api/routers/system.py
printf 'Source backup: %s\nImage backup: %s\nAPI replicas: %s\n' "$backup" "$rollback_tag" "$replicas"
```

Record the printed backup paths, then apply your three-file update and build:

```bash
tar -xzf ~/brain-mri-h5-update.tar.gz -C ~/brain-mri
dc build api
```

Only after the build succeeds, recreate the API service and refresh nginx's
container addresses. This briefly interrupts predictions with one replica.

```bash
dc up -d --no-deps --force-recreate --scale api="$replicas" --wait --wait-timeout 180 api
dc restart gateway
```

This uses the existing VM and replica count. It does not start training or modify
the dataset, registry, uploads, credentials or VM schedule. The UI, probe and
training worker do not need recreation for this API serving change. They may still
use the preceding shared image until a later full-stack update.

## Verify every API replica

```bash
for container in $(dc ps -q api); do
  docker exec "$container" python -c "import json, urllib.request; h=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=15)); print(h); assert h['status']=='ok' and h['model_artifact']=='model.h5'"
done
curl -fsS https://YOUR-SITE-HOST/api/health
```

The public response must show `model_loaded: true`, the expected model version and
`model_artifact: "model.h5"`. Test a known MRI through the public prediction page as
well. A missing artifact field means the request reached older API code.

From the laptop, the existing deployment checker can also exercise prediction:

```powershell
python scripts/check_deployment.py --api https://YOUR-SITE-HOST/api
```

Without upload/retraining flags, this checks health/model information, predicts one
held-out image per class and checks invalid-file handling. It does not start training.

## Roll back if verification fails

In the same VM shell, the variables above refer to the recorded backups:

```bash
tar -xzf "$backup" -C ~/brain-mri
docker image tag "$rollback_tag" brain-mri-api:local
dc up -d --no-deps --force-recreate --scale api="$replicas" --wait --wait-timeout 180 api
dc restart gateway
```

If using a new shell, first restore the recorded values of `backup`, `rollback_tag`
and `replicas`, and define `dc` again. Do not delete model files or Docker volumes
to resolve a serving-format issue.
