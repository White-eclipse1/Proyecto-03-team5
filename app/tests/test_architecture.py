import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent

BANNED_PATTERN = re.compile(
    r"os\.environ|os\.getenv|boto3\.client|Minio\(|create_engine|pymysql\.connect"
)


def test_analyzers_do_not_touch_network_or_env_directly():
    offenders = []
    for path in (APP_ROOT / "analyzers").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if BANNED_PATTERN.search(text):
            offenders.append(str(path))

    assert not offenders, f"Analizadores acoplados a red/entorno: {offenders}"


def test_pyproject_and_dockerfile_pin_python_312():
    pyproject = (APP_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dockerfile = (APP_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'requires-python = "==3.12.*"' in pyproject
    assert "python:3.12-slim" in dockerfile
