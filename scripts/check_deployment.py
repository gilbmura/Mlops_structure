"""Exercise a running deployment end to end through its public API.

Usage:
    python scripts/check_deployment.py --api http://127.0.0.1:8000
    python scripts/check_deployment.py --api http://127.0.0.1:8000 --upload-zip data/cache/retrain_demo_upload.zip --retrain smoke

Checks /health and /model, predicts one held-out test image per class, confirms that
non-image content is refused, and optionally (admin key from ADMIN_API_KEY) uploads a
class-folder ZIP, admits the staged images, starts a retraining job and follows it to
completion. Prints a JSON report; exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from utils.files import read_csv  # noqa: E402


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def check_predictions(api: str, data_dir: Path, failures: list[str]) -> list[dict]:
    rows, seen = [], set()
    for row in read_csv(data_dir / "manifests" / "test.csv"):
        if row["label"] in seen:
            continue
        seen.add(row["label"])
        path = data_dir / row["path"]
        response = requests.post(f"{api}/predict", files={"file": (path.name, path.read_bytes())}, timeout=60)
        _check(response.status_code == 200, f"/predict returned {response.status_code} for {path.name}", failures)
        body = response.json()
        rows.append({"file": path.name, "true": row["label"], "predicted": body.get("predicted_class"),
                     "confidence": body.get("confidence"), "message": body.get("message"), "served_by": response.headers.get("X-Served-By")})
    bad = requests.post(f"{api}/predict", files={"file": ("scan.jpg", b"not an image")}, timeout=30)
    _check(bad.status_code == 422, f"non-image upload returned {bad.status_code}, expected 422", failures)
    return rows


def run_upload_and_retrain(api: str, key: str, archive: Path, mode: str, failures: list[str], timeout: float) -> dict:
    headers = {"X-Admin-Key": key}
    report = requests.post(f"{api}/uploads/zip", headers=headers, timeout=300,
                           files={"file": (archive.name, archive.read_bytes(), "application/zip")})
    _check(report.status_code == 200, f"ZIP upload returned {report.status_code}: {report.text[:200]}", failures)
    batch = report.json()
    if report.status_code != 200:
        return {"upload_status": report.status_code}
    staged = [item["id"] for item in batch.get("items", []) if item["status"] == "staged"]
    result = {"batch_id": batch.get("batch_id"), "upload_counts": batch.get("counts")}
    if staged:
        admitted = requests.post(f"{api}/uploads/admit", headers=headers, timeout=300,
                                 json={"upload_ids": staged, "confirm_labels_reviewed": True})
        _check(admitted.status_code == 200, f"Admission returned {admitted.status_code}", failures)
        if admitted.status_code != 200:
            return result
        result["admitted"] = len(admitted.json()["admitted"])
    if mode == "none":
        return result
    if not result.get("admitted"):
        _check(False, "This upload admitted no images; refusing to retrain an unrelated existing pool.", failures)
        return result

    job = requests.post(f"{api}/jobs/retrain", headers=headers,
                        json={"smoke": mode == "smoke", "batch_id": batch["batch_id"]}, timeout=30)
    _check(job.status_code == 202, f"retrain request returned {job.status_code}: {job.text[:200]}", failures)
    if job.status_code != 202:
        return result
    job_id, deadline, statuses = job.json()["id"], time.monotonic() + timeout, []
    while time.monotonic() < deadline:
        state = requests.get(f"{api}/jobs/{job_id}", timeout=30).json()
        if not statuses or statuses[-1] != state["status"]:
            statuses.append(state["status"])
        health = requests.get(f"{api}/health", timeout=30)
        _check(health.status_code == 200, "API stopped answering /health while training", failures)
        if state["status"] in ("succeeded", "failed"):
            break
        time.sleep(5)
    result.update({"job_id": job_id, "status_sequence": statuses, "final": state})
    _check(state["status"] == "succeeded", f"retraining job ended as {state['status']}: {state.get('error')}", failures)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--upload-zip", type=Path)
    parser.add_argument("--retrain", choices=["none", "smoke", "full"], default="none")
    parser.add_argument("--timeout", type=float, default=7200, help="seconds to wait for the retraining job")
    args = parser.parse_args()
    settings = get_settings()
    api, failures = args.api.rstrip("/"), []

    health = requests.get(f"{api}/health", timeout=30)
    _check(health.status_code == 200 and health.json()["status"] == "ok", f"/health: {health.status_code} {health.text[:200]}", failures)
    report = {"api": api, "health": health.json(), "active_model": requests.get(f"{api}/model", timeout=30).json().get("active_version")}
    report["predictions"] = check_predictions(api, settings.data_dir, failures)
    if args.upload_zip:
        if not settings.admin_api_key:
            sys.exit("Set ADMIN_API_KEY (or .env) to upload and retrain.")
        report["retraining"] = run_upload_and_retrain(api, settings.admin_api_key, args.upload_zip, args.retrain, failures, args.timeout)
        report["active_model_after"] = requests.get(f"{api}/model", timeout=30).json().get("active_version")
    report["failures"] = failures
    print(json.dumps(report, indent=2, default=str))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
