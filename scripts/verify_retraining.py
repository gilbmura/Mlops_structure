"""Verify saved-weight retraining via API upload and a separate local worker.

Runs one full-data epoch in an isolated registry under data/cache. The served
registry is read-only throughout. Saves actual job evidence under reports/.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import yaml

from api.main import create_app
from config.settings import get_settings, load_pipeline_config
from src.dataset.manifests import read_split
from src.registry import ModelRegistry
from utils.files import write_json
from utils.hashing import sha256_file


def main() -> None:
    source = get_settings()
    registry = ModelRegistry(source.models_dir)
    parent = registry.active_version()
    if not parent:
        raise RuntimeError("Activate an initial trained model first.")
    run_dir = source.cache_dir / f"saved_weights_check_{uuid.uuid4().hex[:8]}"
    model_dir = run_dir / "models"
    shutil.copytree(registry.version_dir(parent), model_dir / "versions" / parent)
    write_json(model_dir / "registry.json", {
        "active": parent, "activations": [], "versions": {parent: registry.state()["versions"][parent]},
    })
    settings = dataclasses.replace(source, models_dir=model_dir, state_dir=run_dir / "state")
    config = copy.deepcopy(load_pipeline_config())
    config["training"]["max_epochs"] = 1
    config_path = run_dir / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    parent_path = registry.version_dir(parent) / "model.keras"
    checksum = sha256_file(parent_path)
    sample = read_split(source.data_dir, "retrain_demo")[0]
    headers = {"X-Admin-Key": settings.admin_api_key}
    with TestClient(create_app(settings, config)) as client:
        response = client.post("/uploads/images", headers=headers,
            files={"files": (sample.path.name, sample.path.read_bytes())}, data={"label": sample.label})
        response.raise_for_status()
        item = response.json()["items"][0]
        assert item["status"] == "staged", item
        response = client.post("/uploads/admit", headers=headers,
            json={"upload_ids": [item["id"]], "confirm_labels_reviewed": True})
        response.raise_for_status()
        response = client.post("/jobs/retrain", headers=headers, json={"smoke": False})
        response.raise_for_status()
        job = response.json()
        assert job["status"] == "queued"
        env = {**os.environ, "APP_DATA_DIR": str(source.data_dir), "MODELS_DIR": str(model_dir),
               "APP_STATE_DIR": str(settings.runtime_dir), "PIPELINE_CONFIG": str(config_path),
               "TF_CPP_MIN_LOG_LEVEL": "2"}
        with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "worker.main", "--once"], cwd=ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        finished = client.get(f"/jobs/{job['id']}").json()
        assert finished["status"] == "succeeded", finished
        result = finished["result"]
        assert result["parent_version"] == parent
        assert result["initialization"] == "saved_custom_weights"
        assert sha256_file(parent_path) == checksum
        candidate = ModelRegistry(model_dir).metadata(result["candidate_version"])
        assert candidate["parent_artifact"]["sha256"] == checksum
        evidence = {"scope": "isolated local worker; one epoch, full training and validation splits",
                    "sample_id": sample.sample_id, "upload_status": item["status"],
                    "queued_status": job["status"], "job": finished,
                    "parent_unchanged": True, "candidate_metadata": candidate,
                    "isolated_artifacts": str(model_dir.relative_to(ROOT))}
        destination = source.reports_dir / "retraining_demo" / "saved_weights_check.json"
        write_json(destination, evidence)
        shutil.copyfile(run_dir / "worker.log", destination.with_name("saved_weights_worker.log"))
        print(json.dumps({"evidence": str(destination), "result": result}, indent=2))


if __name__ == "__main__":
    main()
