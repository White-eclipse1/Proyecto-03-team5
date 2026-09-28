import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"


def _write_report(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_version": "test",
                "status": status,
                "checks": [
                    {
                        "check_name": "fixture",
                        "passed": status != "failed",
                        "metric_value": 0.0,
                        "details": {},
                        "action": "fail",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _run_dvc_gate(tmp_path: Path, status: str) -> subprocess.CompletedProcess[str]:
    reports_dir = tmp_path / "reports"
    _write_report(reports_dir / "quality.json", status)
    env = os.environ.copy()
    env["REPORTS_DIR"] = str(reports_dir)
    return subprocess.run(
        [sys.executable, str(APP / "dvc_gate_stage.py")],
        cwd=APP,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_dvc_gate_fails_without_writing_pass_marker(tmp_path):
    result = _run_dvc_gate(tmp_path, "failed")

    assert result.returncode == 1
    assert "BLOQUEADA" in result.stderr
    assert not (tmp_path / "reports" / ".quality_gate.passed").exists()


def test_dvc_gate_passes_and_writes_marker(tmp_path):
    result = _run_dvc_gate(tmp_path, "passed")

    assert result.returncode == 0
    assert (tmp_path / "reports" / ".quality_gate.passed").read_text(encoding="utf-8") == "passed\n"


def test_dvc_pipeline_has_report_gate_and_downstream_split():
    pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = pipeline["stages"]

    assert "always_changed" not in stages["quality_report"]
    assert stages["quality_report"]["cmd"].endswith("dvc_quality_report_stage.py")
    assert stages["quality_gate"]["deps"] == [
        "../reports/quality.json",
        "dvc_gate_stage.py",
        "presentation/contracts.py",
    ]
    assert any(
        output == "../reports/.quality_gate.passed"
        or (
            isinstance(output, dict)
            and (
                output.get("path") == "../reports/.quality_gate.passed"
                or "../reports/.quality_gate.passed" in output
            )
        )
        for output in stages["quality_gate"]["outs"]
    )
    assert "../reports/.quality_gate.passed" in stages["split"]["deps"]
