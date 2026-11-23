"""Locust workload: real MRI image uploads to POST /predict through the load balancer.

Images come from the held-out test split (data/test by default; override with
LOADTEST_IMAGES). Each simulated user uploads a random image, waits 0.5-1.5 s
("think time"), and repeats. The replica that answered (X-Served-By header) is
counted so the run proves how many containers actually served traffic.

    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

IMAGE_DIR = Path(os.environ.get("LOADTEST_IMAGES", "data/test"))
IMAGES = [(path.name, path.read_bytes()) for path in sorted(IMAGE_DIR.rglob("*.jpg"))]
if not IMAGES:
    raise SystemExit(f"No .jpg images found under {IMAGE_DIR}; prepare the dataset first.")

served_by: Counter[str] = Counter()


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predict(self) -> None:
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="POST /predict", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:120]}")
                return
            served_by[response.headers.get("X-Served-By", "unknown")] += 1
            response.success()


@events.spawning_complete.add_listener
def reset_replica_counts(**_kwargs) -> None:
    """Mirror Locust's --reset-stats so ramp-up requests are excluded from replica counts too."""
    served_by.clear()


@events.test_stop.add_listener
def write_replica_counts(environment, **_kwargs) -> None:
    """Save how many successful responses each replica produced (path from LOADTEST_REPLICA_FILE)."""
    target = os.environ.get("LOADTEST_REPLICA_FILE")
    if target:
        Path(target).write_text(json.dumps(dict(served_by), indent=2), encoding="utf-8")
