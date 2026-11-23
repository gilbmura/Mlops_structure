# Local load-test results (generated from raw Locust CSVs)

Machine: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 logical CPUs); Docker VM 8 CPUs / 7.59 GiB.
Each API container: 1.0 CPU, 1536M memory, TF threads 1. Load balancer: nginx least_conn.
Workload: 60 users, think time 0.5-1.5 (uniform) s, warm-up 30 s (excluded), measured 120 s with stats reset after ramp-up. Locust ran on the same machine as the containers; no training job was running.

| API containers | Requests | Failures | Failure rate | Successful req/s | Median ms | p95 ms | p99 ms | Responses per replica |
|---|---|---|---|---|---|---|---|---|
| 1 | 3405 | 0 | 0.00% | 30.41 | 860 | 1800 | 3000 | 3425 |
| 2 | 5240 | 0 | 0.00% | 46.40 | 220 | 740 | 1100 | 2616, 2627 |
| 3 | 6081 | 0 | 0.00% | 53.82 | 58 | 340 | 540 | 2043, 2042, 2016 |
