# Demo video walkthrough plan (target 8–10 minutes)

> YouTube link: **PENDING** (record using the hosted Azure app after verification; add the URL to the README).

| # | Time | Show | Say / point out |
|---|---|---|---|
| 1 | 0:00 | README title + disclaimer | Educational prototype, four classes, derived tumour status, not diagnostic |
| 2 | 0:40 | `data/manifests/dataset_card.json`, notebook §2–3 | 6,000 files verified; 134 exact-duplicate groups (98 across the supplied split) → new group-aware split; near-duplicate thresholds and contact sheet |
| 3 | 2:00 | Notebook §4 | Brightness / contrast / texture by class, and the acquisition confound (non-square, non-512 sources concentrated in *no_tumor*) |
| 4 | 3:00 | Notebook §6–8 | Baseline vs CNN on validation; final test metrics once; confusion matrix; shortcut check |
| 5 | 4:15 | Azure VM: Compose services using both base and Azure files | Architecture: Caddy → Streamlit/nginx → API replicas; worker trains outside requests |
| 6 | 5:00 | Hosted UI *Predict* | Upload a test image; message, confidence + disclaimer; upload a photo → refused ("doesn't look like a brain MRI slice"), no class shown |
| 7 | 6:00 | Hosted UI *Uploads & retraining* | Show initial-training controls; single image with label; ZIP with one class folder; a test-set image rejected for leakage; review + admit |
| 8 | 7:00 | Hosted retraining (quick demo run), then *System status* | Job queued → running → succeeded/failed; prediction still answers while training; promotion decision and rollback button |
| 9 | 8:00 | Actual Azure Locust results | 1 vs 2 vs 3 containers: throughput, failures, p95/p99 — explain resource limits on the VM |
| 10 | 9:00 | Live Azure site and production evidence | Show the updated deployment, health, prediction and cloud load-test results after verifying them; local evidence is not cloud evidence |

Recording tips: keep the admin key off-screen and use the *quick demo run* so retraining can
finish during the video. For scaling on the VM, retain both Compose files as documented in
`deploy/azure/README.md`; collect cloud measurements before recording.

## Rubric evidence

Keep your camera on and verify microphone audio. Use a documented correctly classified image for
prediction and state its true label. Show the uploaded file and its SQLite path/label/review record,
shared preprocessing, and a retraining job that loads the project's saved CNN. Explain that initial
training used random weights while retraining continues those learned weights with a fresh optimizer.
Show parent_version, initialization, validation results and the promotion decision in the completed job.
Quick runs are genuine training checks but never promote and use a subset; label them clearly.
Historical v003 results used the previous from-scratch retraining workflow.

## Azure recording setup

Use https://gilbert-brain-mri-demo.southafricanorth.cloudapp.azure.com for the demonstration.
The separate local recording app has been removed. Keep the main local app for development.

The original 100 reserved images remain in `data/retrain_demo/<class>/`, with the existing
class-folder archive at `data/cache/retrain_demo_upload.zip`. They are excluded from initial
training. Having these files on Azure does not automatically admit them to its upload pool.
Azure and the local app have independent upload databases; check the Azure database first.

After starting the VM, SSH into it and run the following **read-only** check in `~/brain-mri`.
It compares decoded-image fingerprints, so renaming a previously uploaded file does not hide
a duplicate. It does not require revealing the admin key or uploading anything.

```bash
cd ~/brain-mri
docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml exec -T api python - <<'PY'
import json
import sqlite3
from collections import Counter
import numpy as np

with np.load('/app/data/cache/similarity_index.npz', allow_pickle=False) as index:
    reserved = set(index['pixel_hashes'][index['splits'] == 'retrain_demo'].tolist())
with sqlite3.connect('file:/app/state/app.db?mode=ro', uri=True) as db:
    rows = db.execute('SELECT status, pixel_hash FROM uploads').fetchall()
kept = {h for status, h in rows if status in ('staged', 'needs_label', 'admitted')}
admitted = {h for status, h in rows if status == 'admitted'}
print(json.dumps({
    'upload_counts': dict(Counter(status for status, _ in rows)),
    'reserved_images': len(reserved),
    'reserved_already_kept': len(reserved & kept),
    'reserved_already_admitted': len(reserved & admitted),
    'reserved_not_in_upload_pool': len(reserved - kept),
}, indent=2))
PY
```

If `reserved_already_kept` is zero, this upload-history check found no existing copies in the
pool. The normal content and leakage checks still apply when uploading. If it is 100, all
reserved images are already staged or admitted and uploading them again will be skipped as
duplicates. A value between 0 and 100 means some remain unused. Use distinct images for the
single, multiple and ZIP demonstrations. Review pending images or demonstrate retraining from
the existing admitted pool when appropriate; do not delete production history or weaken checks
just to make previously uploaded images appear new.

Show **Train a new model from scratch** as well as upload/retraining controls after deploying
the update. Full initial-training candidates require manual activation after validation review;
quick checks cannot be activated.

## Before the final recording

- Deploy the current local changes to Azure and verify the UI, `/api/health`, `/api/docs`,
  upload screening and both retraining actions. The VM was confirmed deallocated on 2026-11-24;
  it was left stopped at the owner's request.
- Run the reserved-image check above on Azure. The local pool's 100 admitted images do not
  establish what is in the cloud database. Azure upload history has not yet been inspected.
- Show that a rejected batch leaves **Retrain with this batch** disabled even when the separate
  **Retrain with existing pool** action is available. Accepted images require review and admission.
- Use a small class-folder ZIP, not the complete BRISC archive (200 files per upload).
- Keep a completed full training job available to explain validation and promotion. A quick
  demo job is a subset run and cannot promote, so do not present it as a full retraining result.
- Show at least three feature interpretations from the notebook. Include the known classification
  errors, patient-identity limitation and the limits of the non-MRI filter.
- Capture cloud Locust evidence at different replica counts and the production evaluation
  result. Add the YouTube link and current deployment URL to the main README before submission.
