# Demo video walkthrough plan (target 8–10 minutes)

> YouTube link: **PENDING** (record after the final local run; add the URL to the README).

| # | Time | Show | Say / point out |
|---|---|---|---|
| 1 | 0:00 | README title + disclaimer | Educational prototype, four classes, derived tumour status, not diagnostic |
| 2 | 0:40 | `data/manifests/dataset_card.json`, notebook §2–3 | 6,000 files verified; 134 exact-duplicate groups (98 across the supplied split) → new group-aware split; near-duplicate thresholds and contact sheet |
| 3 | 2:00 | Notebook §4 | Brightness / contrast / texture by class, and the acquisition confound (non-square, non-512 sources concentrated in *no_tumor*) |
| 4 | 3:00 | Notebook §6–8 | Baseline vs CNN on validation; final test metrics once; confusion matrix; shortcut check |
| 5 | 4:15 | `docker compose ps` (3 API replicas, gateway, worker, probe, ui) | Architecture: Streamlit → nginx → API replicas; worker trains outside requests |
| 6 | 5:00 | UI *Predict* | Upload a test image; message, confidence + disclaimer; upload a photo → refused ("doesn't look like a brain MRI slice"), no class shown |
| 7 | 6:00 | UI *Uploads & retraining* | Admin key; single image with label; ZIP with one class folder; a test-set image rejected for leakage; review + admit |
| 8 | 7:00 | Start retraining (quick demo run), then *System status* | Job queued → running → succeeded/failed; prediction still answers while training; promotion decision and rollback button |
| 9 | 8:00 | `loadtest/results/SUMMARY.md` | 1 vs 2 vs 3 containers: throughput, failures, p95/p99 — and why gains are limited on one laptop |
| 10 | 9:00 | `deploy/azure/README.md` | Azure plan ready; cloud URL and cloud load test still pending |

Recording tips: run `docker compose up -d --scale api=3` beforehand, keep the admin key
off-screen, and use the *quick demo run* so retraining finishes during the video.

## Rubric evidence

Keep your camera on and verify microphone audio. Use a documented correctly classified image for
prediction and state its true label. Show the uploaded file and its SQLite path/label/review record,
shared preprocessing, and a retraining job that loads the project's saved CNN. Explain that initial
training used random weights while retraining continues those learned weights with a fresh optimizer.
Show parent_version, initialization, validation results and the promotion decision in the completed job.
Quick runs are genuine training checks but never promote and use a subset; label them clearly.
Historical v003 results used the previous from-scratch retraining workflow.
