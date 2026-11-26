# Azure deployment and verification

> The stack is deployed at https://gilbert-brain-mri-demo.southafricanorth.cloudapp.azure.com.
> HDF5 prediction serving was deployed and verified on 2026-11-26 with one API replica.
> See [the update and rollback record](HDF5_UPDATE.md) and
> [public health/prediction evidence](../../reports/deployment/hdf5_20261126.json).
> Availability depends on the VM's running schedule. The project author confirms
> that production evaluation and the 1/2/3-replica performance comparison are
> demonstrated in the [video](https://youtu.be/7QIUv2JxGkg). Supplementary reports
> are in `loadtest/results/azure_final/`.

## Recommended target: one Linux VM running the same Docker Compose stack

| Need | How it is met |
|---|---|
| UI + API | Same images as local; Caddy terminates HTTPS and routes `/` → Streamlit, `/api/*` → nginx → API replicas |
| Persistent uploads / models | VM managed disk: `./data`, `./models`, `./reports` on disk; SQLite + staged uploads on the `state` Docker volume |
| Background retraining | `worker` container (one job at a time; DB refuses concurrent retrain jobs) |
| Controlled Docker scaling | `docker compose up -d --scale api=N` then `docker compose restart gateway` — identical to the local experiment |
| Health checks | Container `HEALTHCHECK`s, `/health` (per replica), `probe` service records measured availability |
| Admin protection | `ADMIN_API_KEY` required for uploads, retraining, promotion, rollback; optional Caddy `basic_auth` for the whole site; SSH limited to your IP |
| HTTPS | Caddy + Let's Encrypt for `<dns-label>.<region>.cloudapp.azure.com` (ports 80/443 only) |
| Streamlit UI | The `ui` container on the same VM; Caddy serves it at `/` (websockets included). It calls the API privately at `http://gateway:8000`, never over the internet |
| Swagger under `/api` | Caddy strips the `/api` prefix; the Azure override sets `UVICORN_ROOT_PATH=/api` so `/api/docs` loads `/api/openapi.json` and "Try it out" calls `/api/...` |

**Why a VM instead of Azure Container Apps / App Service?** The stack keeps SQLite and
uploaded files on a shared filesystem. Container Apps would put them on Azure Files (SMB),
where SQLite locking is unreliable, and would need a managed database plus blob storage —
more services, more cost, and a setup that differs from what was tested locally. A single
VM reproduces the tested architecture exactly. The trade-off: the VM is a single point of
failure and scaling is limited to that machine's CPUs, so it demonstrates container
scaling, not high availability.

## Cost control (student subscription)

* Check the current hourly price of the chosen size (default `Standard_B2as_v2`) for your region
  in the [Azure pricing calculator](https://azure.microsoft.com/pricing/calculator/) before deploying;
  prices are not hard-coded here because they change and vary by region.
* Create a **budget with e-mail alerts** (Cost Management → Budgets) well below the $100 credit.
* `provision.ps1` sets a **daily auto-shutdown**. A *stopped (deallocated)* VM stops compute billing;
  the disk and public IP still cost a little.
* Stop when idle: `az vm deallocate -g rg-brain-mri -n vm-brain-mri` (start again with `az vm start`).
* Remove everything when the assessment is over: `az group delete -n rg-brain-mri` (irreversible).
* Retraining on 2 vCPUs is slower than locally; prefer the *quick demo run* for live demonstrations.
* The Azure override caps the worker at one CPU by default (`AZURE_WORKER_CPUS` overrides it),
  rather than inheriting the local four-CPU limit. Leave CPU headroom for prediction during training.

## Steps (when you decide to deploy)

1. **Licensing first.** Confirm the BRISC 2025 licence permits hosting the data/model publicly
   (see the main README, *Dataset licence and attribution*). Until then, keep the site behind
   `basic_auth` or do not make it public.
2. **Dry run:** `./deploy/azure/provision.ps1 -DnsLabel <unique-label> -AllowedSshIp <your-ip>`;
   review the printed `az` commands. Then `az login` and re-run with `-Execute`.
3. **Copy code, data and model** (from your machine; the dataset goes only to your own VM):
   ```bash
   ssh azureuser@<label>.<region>.cloudapp.azure.com 'git clone <your-repo-url> brain-mri'
   scp "archive (6).zip" azureuser@<host>:~/brisc.zip
   scp -r models azureuser@<host>:~/brain-mri/      # or rely on the tracked delivered model
   ```
4. **Prepare data on the VM** (inside a throw-away container using the API image):
   ```bash
   cd ~/brain-mri && cp .env.example .env    # set ADMIN_API_KEY (new, strong) and SITE_HOST
   docker compose build
   docker run --rm -v "$PWD:/app/repo" -v ~/brisc.zip:/data/brisc.zip:ro -w /app/repo brain-mri-api:local \
       python scripts/prepare_dataset.py --archive /data/brisc.zip
   ```
5. **Start with HTTPS:**
   `docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml up -d`
6. **Verify:** `https://<host>/` (UI), `https://<host>/api/health`, `https://<host>/api/docs` (Swagger);
   check the *System status* page shows the probe's measured availability.

**Optional rehearsal on your own computer first** (no Azure, no cost): run the same configuration with
Caddy on `https://localhost:8443` using `deploy/azure/docker-compose.rehearsal.yml` (commands inside the
file). Rehearsed on 2026-11-22: `/api/health` OK; Swagger pointed at `/api/openapi.json` and the spec
listed `/api` as its server; MRI accepted and photo refused via `/api/predict`; Streamlit page and its
websocket worked through Caddy; the UI's and probe's internal calls (no prefix) worked; ports 8000 and
8501 were closed from outside. Browsers show a certificate warning locally because Caddy's local
certificate is not publicly trusted; on Azure, Let's Encrypt provides a trusted one.

## Reproducing the cloud load test

On the VM, select the API replica count. Repeat with 1, 2 and 3 using the same
load-test settings, with no training job running during measurements:

```bash
docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml up -d --scale api=1
docker compose -f docker-compose.yml -f deploy/azure/docker-compose.azure.yml restart gateway
```

From the laptop, open Locust against the bare HTTPS origin. The workload already
includes `/api/predict`, so do not append `/api` to the host:

```powershell
python -m locust -f loadtest/locustfile.py --host https://YOUR-SITE-HOST
```

Open `http://localhost:8089`, configure the same users, spawn rate and duration for
each run, then stop and download its reports before starting the next. Store final
cloud evidence under `loadtest/results/azure_final/{1,2,3}_replicas/`.

Record: VM size and vCPUs, per-container limits, replicas, users, request rate, successful
throughput, failure rate, median/p95/p99 latency, warm-up and duration — and note that client
network latency is included. Results here will differ from the local ones; do not reuse local numbers.

| Item | Status |
|---|---|
| Azure public URL | https://gilbert-brain-mri-demo.southafricanorth.cloudapp.azure.com (current reachability unverified) |
| Cloud health / sample predictions | HDF5 serving verified 2026-11-26; see `reports/deployment/hdf5_20261126.json` |
| Production model evaluation | Demonstrated in the [video](https://youtu.be/7QIUv2JxGkg), confirmed by the project author |
| Cloud load test (1/2/3 replicas) | Comparison demonstrated in the [video](https://youtu.be/7QIUv2JxGkg); [supplementary stress-test and routing evidence](../../loadtest/results/azure_final/README.md) attached |
