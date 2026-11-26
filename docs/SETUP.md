# Set up your own installation

Run commands from the repository root. Use your own virtual environment, admin key,
runtime database and, if deploying, cloud account. The submitted model can be used
immediately for inference; preparing the original dataset is required for the full
training, upload screening and retraining workflows.

## 1. Install the Python environment

The project declares Python 3.12 or 3.13; the recorded local runs used Python 3.13.
On Windows, with Python 3.13 installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in the commands below. Activation is a convenience, not a requirement.

On Linux with Python 3.13 and its `venv` support installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The requirements include the CPU training stack, Jupyter, API, UI, tests and Locust.
Docker is optional for running locally; install Docker with Compose if using step 5B.

## 2. Generate your own admin key

For a **fresh checkout**, create the private configuration file:

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

On Linux, use `cp .env.example .env` for the first command. Do not overwrite an
existing `.env` when updating an installation.

Paste the generated value after `ADMIN_API_KEY=` in `.env`. Leave the local API
address as `API_URL=http://127.0.0.1:8000`. Keep the key out of Git, ZIP submissions,
screenshots and videos. Each installation should generate its own key; the example
file intentionally contains no working key.

Enter that same value in the UI's **Admin key** field for uploads and training controls.
API clients send it as `X-Admin-Key`. Prediction does not require this key. Offline
training scripts run directly on your machine and do not authenticate through the API.
Restart running services after changing `.env`; Docker containers need recreation to
receive changed environment variables.

## 3. Prepare the original dataset

Download the BRISC 2025 archive from the [dataset source](https://www.kaggle.com/datasets/briscdataset/brisc2025).
Preserve its original structure and manifests. See the main README for attribution
and the recorded source limitations. Store the downloaded ZIP outside this repository.

```powershell
python scripts/prepare_dataset.py --archive "C:\path\to\brisc2025.zip"
python scripts/analyze_features.py
```

On Linux, substitute the archive's Linux path. This creates all four splits and
`data/cache/similarity_index.npz`, which the upload service needs for duplicate and
evaluation-leakage checks. The checked-in train/test images alone are insufficient
for the complete workflow. Do this before starting services for the first time.

For the recorded archive and configuration, the cleaned splits contain 4,020 training,
868 validation, 868 test and 100 reserved retraining-demo images. The demo images are
under `data/retrain_demo/<class>/`; keep final-test images out of retraining.

## 4. Choose the supplied model or train your own

To use the supplied model, retain `models/registry.json` and all model directories it
references, then continue to step 5. The submitted active model is `v002`.

To train your own model **from random initial weights**:

```powershell
python scripts/train_model.py --smoke
python scripts/train_model.py --architecture baseline
python scripts/train_model.py --architecture cnn --activate
python scripts/evaluate_model.py --split validation
```

The smoke check uses a small subset and a separate cache registry. The baseline is
optional for running the app but useful for comparing models in the assignment.
The full CNN command creates a new version and makes it active. It does not reuse
the supplied model's weights, and you do not need to delete existing model folders.

After choosing the final model using validation results, evaluate it on the test split:

```powershell
python scripts/evaluate_model.py --split test
```

An existing final-test report is protected against accidental overwrite. For the
supplied model, read the existing reports instead of repeatedly evaluating it while
tuning. New results are saved under `reports/models/<new-version>/`.

The UI's **Train a new model from scratch** action also starts with random weights.
**Retraining** continues the project's own saved custom CNN using approved uploads;
it is a separate workflow from initial training. No external pretrained model is used.
The top-level `models/brain_mri_classifier.h5` is a submission copy of `v002` and is
not updated automatically when you activate another version; see `models/README.md`.

## 5A. Run without Docker

Open four terminals in this repository, activate the same environment in each, and
run one command in each terminal:

```powershell
python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000
```

```powershell
python -m worker.main
```

```powershell
python -m worker.probe --interval 30
```

```powershell
python -m streamlit run ui/app.py
```

Open [the UI](http://localhost:8501) or [the API documentation](http://127.0.0.1:8000/docs).
The worker must be running for queued ZIP processing and training jobs to complete.
The probe records availability for the system-status page. Runtime state is created
under `data/runtime/`; another user's upload or job database is not needed.

## 5B. Run with Docker instead

Complete dataset preparation on the host first, then start Docker and run:

```powershell
docker compose up -d --build
docker compose ps
```

Use the same local UI and API links as above. The stack includes the API, UI, worker,
probe and nginx gateway. Compose supplies the internal API URL automatically.
Do not run the host services and this stack on the same ports at the same time.

Docker stores uploads and job history in its `state` volume. Copying the repository
does not copy that volume, but another checkout on the **same Docker host**, using
the same Compose project name, can reuse it. Do not delete a working installation's
volume just to prepare a submission.

## 6. Check the workflow and notebook

1. Upload an MRI slice on **Predict** and inspect the class and model score.
2. On **Uploads & retraining**, enter your admin key and upload reserved demo images
   as single files, multiple files or a ZIP containing class folders.
3. Review acceptable images and their labels, then admit them. Rejected images do
   not enter training. A mixed batch can use its admitted images; a batch with zero
   admitted images cannot enable batch retraining.
4. Start a retraining job and follow its status. Registration does not guarantee
   promotion: the candidate must satisfy the validation checks.

To inspect the executed notebook:

```powershell
python -m jupyter notebook notebook/brain_mri_classifier.ipynb
```

Use this environment's kernel. Reading saved outputs does not rerun training.
Executing all cells runs a small training smoke check even with `RUN_FULL_TRAINING`
set to `False`; full training is opt-in. Preserve the submitted outputs as evidence
before experimenting. The notebook uses the recorded `v001`/`v002` model comparison.

## 7. Deploy and test in your own cloud account

Use [the Azure deployment guide](../deploy/azure/README.md) with your own subscription,
DNS label, current client IP and SSH keys. Create a separate `.env` on your VM with
a newly generated admin key and `SITE_HOST=<your-VM-DNS-name>`. The author's public
demo address is not a resource assigned to a fresh installation.

The provisioning script defaults to a dry run. Adding `-Execute` creates Azure
resources that consume credit. Its shutdown time is configurable with
`-AutoShutdownUtc`; it does not provision an automatic daily startup schedule.
Review the printed configuration and use the existing single-VM setup for container
comparisons before considering additional cloud resources.

The current Locust workload targets Azure's `/api/predict` endpoint. From your laptop:

```powershell
python -m locust -f loadtest/locustfile.py --host https://YOUR-SITE-HOST
```

Open [Locust](http://localhost:8089), use the bare HTTPS origin without appending `/api`,
and run against an installation you own. Keep the workload and duration comparable
for 1, 2 and 3 API replicas. Stop each run and download its reports before starting
the next one. Save final evidence as described in
[the Azure results folder](../loadtest/results/azure_final/README.md).

**Known local load-test mismatch:** `run_scaling.py` targets the local gateway, whose
prediction path is `/predict`, while `locustfile.py` currently posts to `/api/predict`.
The local helper needs those routes aligned before it is reused. This does not
change the Azure command above or the previously recorded local measurements.
