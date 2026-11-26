# Final Azure load-test evidence

Add the final measured reports here before submission. This folder currently
contains instructions only; no missing result has been inferred or manufactured.
The earlier local measurements remain in the sibling `1_replicas/`, `2_replicas/`
and `3_replicas/` folders. Do not present those laptop measurements as Azure results.

Use these locations for the final cloud runs:

```text
loadtest/results/azure_final/
  1_replicas/
  2_replicas/
  3_replicas/
```

For each run, save the Locust HTML report and downloaded statistics/history/failure
CSVs when available, plus the resource-usage recording and evidence of which API
containers served requests. Preserve the actual measurements and label warm-up
output separately.

Record the date, VM size, API replica count, users, spawn rate, duration, endpoint,
image set, model version and load-generator location. Compare throughput, failure
rate, median response time, p95 and p99 under the same workload. If a run used
different conditions, state that rather than claiming a controlled comparison.

Add a comparison and interpretation to the main README once all three runs are
present. Public hostnames and container identifiers are useful context; admin keys,
private keys, tokens and full private configuration dumps must not appear in reports.
