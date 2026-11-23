"""Run the 1 / 2 / 3 API-replica Locust experiment against the local Docker stack.

Usage (stack already built; no training job should be running):
    python loadtest/run_scaling.py --replicas 1 2 3 --users 60 --spawn-rate 10 --duration 120 --warmup 30

For each replica count N:
  1. ``docker compose up -d --scale api=N`` and restart the gateway so nginx sees every replica;
  2. wait until N distinct replicas answer /health through the gateway;
  3. warm-up run (kept in warmup/, excluded from the summary);
  4. measured run with --reset-stats (ramp-up excluded), sampling ``docker stats`` meanwhile;
  5. keep Locust's raw CSVs, per-replica response counts and resource samples.
Finally loadtest/results/summary.json and SUMMARY.md are written from the raw files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, UTC
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "loadtest" / "results"
HOST = "http://127.0.0.1:8000"


def run(*args: str, check: bool = True) -> str:
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{completed.stdout}\n{completed.stderr}")
    return completed.stdout


def scale(replicas: int) -> None:
    run("docker", "compose", "up", "-d", "--scale", f"api={replicas}", "--no-build")
    run("docker", "compose", "restart", "gateway")


def wait_for_replicas(replicas: int, timeout: float = 240) -> list[str]:
    deadline = time.monotonic() + timeout
    seen: set[str] = set()
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{HOST}/health", timeout=5)
            if response.status_code == 200:
                seen.add(response.json()["served_by"])
        except requests.RequestException:
            pass
        if len(seen) >= replicas:
            return sorted(seen)
        time.sleep(0.3)
    raise RuntimeError(f"Only {len(seen)} of {replicas} replicas answered within {timeout}s: {seen}")


def sample_docker_stats(target: Path, stop: threading.Event, every: float = 15.0) -> None:
    with target.open("w", encoding="utf-8") as handle:
        while not stop.wait(every):
            output = run("docker", "stats", "--no-stream", "--format", "{{json .}}", check=False)
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            for line in output.splitlines():
                if "brain-mri" in line:
                    handle.write(json.dumps({"at": stamp, **json.loads(line)}) + "\n")


def locust(prefix: Path, users: int, spawn_rate: float, seconds: int, replica_file: Path, reset_stats: bool) -> None:
    command = [
        sys.executable, "-m", "locust", "-f", str(ROOT / "loadtest" / "locustfile.py"), "--headless", "--host", HOST,
        "-u", str(users), "-r", str(spawn_rate), "-t", f"{seconds}s", "--csv", str(prefix), "--stop-timeout", "15", "--only-summary",
    ]
    if reset_stats:
        command.append("--reset-stats")
    env = {**os.environ, "LOADTEST_REPLICA_FILE": str(replica_file), "LOADTEST_IMAGES": str(ROOT / "data" / "test")}
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    console = (completed.stdout + completed.stderr).replace(socket.gethostname(), "<host>")  # keep the machine name out of results
    (prefix.parent / f"{prefix.name}_console.txt").write_text(console, encoding="utf-8")
    if completed.returncode not in (0, 1):  # locust exits 1 when any request failed; that is a result, not a crash
        raise RuntimeError(f"Locust crashed (exit {completed.returncode}); see {prefix.name}_console.txt")


def aggregated_row(stats_csv: Path) -> dict:
    with stats_csv.open(newline="", encoding="utf-8") as handle:
        return next(row for row in csv.DictReader(handle) if row["Name"] == "Aggregated")


def environment_info(args: argparse.Namespace) -> dict:
    docker = json.loads(run("docker", "info", "--format", "{{json .}}"))
    cpu = run("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name", check=False).strip() if os.name == "nt" else platform.processor()
    return {
        "host_cpu": cpu,
        "host_logical_cpus": os.cpu_count(),
        "host_os": platform.platform(),
        "docker_vm_cpus": docker.get("NCPU"),
        "docker_vm_memory_gib": round(docker.get("MemTotal", 0) / 1024**3, 2),
        "api_container_limits": {"cpus": os.environ.get("API_CPUS", "1.0"), "memory": "1536M", "tf_threads": os.environ.get("API_TF_THREADS", "1")},
        "load_balancer": "nginx least_conn",
        "workload": {
            "endpoint": "POST /predict (multipart JPEG from the held-out test split)",
            "users": args.users, "spawn_rate_per_s": args.spawn_rate, "think_time_s": "0.5-1.5 (uniform)",
            "measured_seconds": args.duration, "warmup_seconds": args.warmup,
            "note": "Locust ran on the same machine as the containers; no training job was running.",
        },
    }


def summarize(rows: list[dict], info: dict) -> None:
    (RESULTS / "summary.json").write_text(json.dumps({"environment": info, "runs": rows}, indent=2), encoding="utf-8")
    lines = [
        "# Local load-test results (generated from raw Locust CSVs)", "",
        f"Machine: {info['host_cpu']} ({info['host_logical_cpus']} logical CPUs); Docker VM {info['docker_vm_cpus']} CPUs / {info['docker_vm_memory_gib']} GiB.",
        f"Each API container: {info['api_container_limits']['cpus']} CPU, {info['api_container_limits']['memory']} memory, TF threads {info['api_container_limits']['tf_threads']}. Load balancer: {info['load_balancer']}.",
        f"Workload: {info['workload']['users']} users, think time {info['workload']['think_time_s']} s, warm-up {info['workload']['warmup_seconds']} s (excluded), "
        f"measured {info['workload']['measured_seconds']} s with stats reset after ramp-up. {info['workload']['note']}", "",
        "| API containers | Requests | Failures | Failure rate | Successful req/s | Median ms | p95 ms | p99 ms | Responses per replica |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['replicas']} | {r['requests']} | {r['failures']} | {r['failure_rate']:.2%} | {r['successful_rps']:.2f} | "
            f"{r['median_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {', '.join(str(v) for v in r['responses_per_replica'].values())} |"
        )
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--spawn-rate", type=float, default=10)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    info = environment_info(args)
    rows = []
    for replicas in args.replicas:
        folder = RESULTS / f"{replicas}_replicas"
        (folder / "warmup").mkdir(parents=True, exist_ok=True)
        print(f"== {replicas} replica(s): scaling and waiting for health")
        scale(replicas)
        names = wait_for_replicas(replicas)
        print(f"   replicas answering: {names}; warm-up {args.warmup}s")
        locust(folder / "warmup" / "locust", args.users, args.spawn_rate, args.warmup, folder / "warmup" / "replicas.json", reset_stats=False)

        print(f"   measured run {args.duration}s")
        stop = threading.Event()
        sampler = threading.Thread(target=sample_docker_stats, args=(folder / "docker_stats.jsonl", stop))
        sampler.start()
        started = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            locust(folder / "locust", args.users, args.spawn_rate, args.duration, folder / "replicas.json", reset_stats=True)
        finally:
            stop.set()
            sampler.join()

        row = aggregated_row(folder / "locust_stats.csv")
        requests_total, failures = int(row["Request Count"]), int(row["Failure Count"])
        per_replica = json.loads((folder / "replicas.json").read_text(encoding="utf-8"))
        rows.append({
            "replicas": replicas,
            "replicas_that_served_traffic": len(per_replica),
            "started_at": started,
            "requests": requests_total,
            "failures": failures,
            "failure_rate": failures / requests_total if requests_total else 0.0,
            "requests_per_s": float(row["Requests/s"]),
            "successful_rps": float(row["Requests/s"]) * (1 - failures / requests_total) if requests_total else 0.0,
            "median_ms": float(row["50%"]),
            "p95_ms": float(row["95%"]),
            "p99_ms": float(row["99%"]),
            "mean_ms": float(row["Average Response Time"]),
            "responses_per_replica": per_replica,
        })
        print(f"   {rows[-1]}")
    summarize(rows, info)
    print((RESULTS / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
