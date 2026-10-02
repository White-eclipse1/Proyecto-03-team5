"""OPS-05: smoke training corto contra el stack real de Proyecto 3.

Requiere que `docker compose up --build` esté levantado y que el release
v0.1.1/crops estén disponibles. No requiere GPU.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("OPS05_API_BASE", "http://localhost:8080/api/ml").rstrip("/")
DATASET_VERSION = os.environ.get("OPS05_DATASET_VERSION", "v0.1.1")
TIMEOUT_SECONDS = int(os.environ.get("OPS05_TIMEOUT_SECONDS", "900"))

ROOT = Path(__file__).resolve().parents[2]


def _request_json(
    method: str,
    path: str,
    body: dict | None = None,
) -> tuple[int, dict]:
    data = None
    headers: dict[str, str] = {}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {path} respondio HTTP {exc.code}: {payload}") from exc
    except OSError as exc:
        raise RuntimeError(
            f"No se pudo conectar a {API_BASE}. Verifica que docker compose este levantado."
        ) from exc


def main() -> None:
    manifest_path = ROOT / "reports" / "releases" / DATASET_VERSION / "manifest.json"

    if not manifest_path.is_file():
        raise SystemExit(f"No existe {manifest_path}. Recupera/genera el release antes del smoke.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    print(f"API: {API_BASE}")
    print(f"Dataset: {DATASET_VERSION}")
    print("Smoke training: CPU-compatible, 1 epoca, image_size=32")

    status, health = _request_json("GET", "/health")
    if status != 200 or health.get("status") != "ok":
        raise SystemExit(f"ml-api no esta saludable: {health}")

    request_body = {
        "schema_version": "1.0",
        "dataset_version": DATASET_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "params": {
            "optimizer": "adam",
            "batch_size": 128,
            "max_epochs": 1,
            "learning_rate": 0.001,
            "image_size": 32,
            "hidden_layers": [],
            "dropout": 0.0,
            "seed": 42,
            "patience": 1,
            "min_delta": 0.0,
        },
    }

    status, job = _request_json("POST", "/training/jobs", request_body)

    if status != 202:
        raise SystemExit(f"No se pudo crear el job: HTTP {status}: {job}")

    job_id = job["job_id"]
    print(f"Job creado: {job_id}")

    deadline = time.monotonic() + TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        _, current = _request_json("GET", f"/training/jobs/{job_id}")
        state = current["status"]

        progress = current.get("progress")
        if progress:
            print(
                f"Estado={state} epoca={progress.get('epoch')} metricas={progress.get('metrics')}"
            )
        else:
            print(f"Estado={state}")

        if state == "succeeded":
            run_id = current.get("run_id")
            checkpoint = current.get("checkpoint")

            if not run_id:
                raise SystemExit("Job succeeded pero no tiene run_id.")
            if checkpoint != f"runs:/{run_id}/checkpoints/best.pt":
                raise SystemExit(f"Checkpoint inesperado: {checkpoint!r}")

            print("")
            print("OPS-05 SMOKE OK")
            print(f"job_id={job_id}")
            print(f"run_id={run_id}")
            print(f"checkpoint={checkpoint}")
            return

        if state == "failed":
            raise SystemExit(f"Training fallo: {current.get('error')}")

        time.sleep(2)

    raise SystemExit(f"Timeout tras {TIMEOUT_SECONDS}s esperando el job {job_id}.")


if __name__ == "__main__":
    main()
