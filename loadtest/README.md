# Load testing

`locustfile.py` uploads real JPEG slices from the held-out test split to `POST /predict`
through the nginx load balancer; `run_scaling.py` repeats the same workload with 1, 2 and 3
API containers and writes everything under `results/`:

```
results/<N>_replicas/locust_stats.csv, locust_stats_history.csv, locust_failures.csv   raw Locust output (measured run)
results/<N>_replicas/replicas.json        successful responses counted per container (X-Served-By header)
results/<N>_replicas/docker_stats.jsonl   CPU / memory of every container, sampled every 15 s
results/<N>_replicas/warmup/              warm-up run (excluded from the summary)
results/summary.json, SUMMARY.md          generated from the raw files
```

## Conditions (run on 2026-11-21)

* Host: Intel i5-1135G7 (4 cores / 8 threads), 15.6 GB RAM, Windows 11; Docker Desktop VM with 8 CPUs / 7.6 GiB.
* Each API container: **1.0 CPU**, 1.5 GB memory limit, TensorFlow intra/inter-op threads = 1, one uvicorn worker.
* nginx `least_conn`; Locust 2.47 on the **same machine** (it competes for CPU with the containers).
* 60 users, spawn rate 10/s, think time 0.5–1.5 s; 30 s warm-up discarded, then 120 s measured with
  statistics reset after ramp-up. No training job ran during these measurements.

## Results (from `results/SUMMARY.md`)

| API containers | Requests | Failures | Successful req/s | Median | p95 | p99 | API container CPU (mean of samples) |
|---|---|---|---|---|---|---|---|
| 1 | 3,405 | 0 (0 %) | 30.4 | 860 ms | 1,800 ms | 3,000 ms | 100 % (at its 1-CPU limit) |
| 2 | 5,240 | 0 (0 %) | 46.4 | 220 ms | 740 ms | 1,100 ms | 101 %, 99 % (both at limit) |
| 3 | 6,081 | 0 (0 %) | 53.8 | 58 ms | 340 ms | 540 ms | 84 %, 78 %, 81 % (headroom) |

Responses per container were balanced (e.g. 2,043 / 2,042 / 2,016 with three). These per-container
counts are collected client-side and differ from Locust's totals by under 1 % because the two are reset at
slightly different moments after ramp-up.

## Interpretation

* **One container is CPU-bound**: it sits at its 1-CPU limit and requests queue, so median latency is
  860 ms and p99 3 s.
* **Two containers** raise successful throughput by 53 % and cut median latency about 4×; both are still at
  their CPU limit.
* **Three containers** have spare CPU. Throughput rises only another 16 % because the *workload* is now the
  limit: 60 users with ~1 s think time can generate at most ~56 req/s when responses are fast. The
  clearer gain at this step is latency (median 58 ms, p99 540 ms).
* These numbers describe one laptop, where containers, Locust and the OS share 8 logical CPUs. More
  replicas than free cores would not help, and cloud results will differ (network latency, different vCPUs).
  The Azure measurement is still **pending**.
