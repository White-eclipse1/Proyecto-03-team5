"""OPS-07 — publicación del paquete del modelo en AWS S3 y recuperación en un entorno limpio.

`publish` sube el paquete de OPS-06 de una model version (`checkpoint/best.pt`,
`model-card.md`, `package.json`, `dependencies.json`) a
`s3://<bucket>/<prefix>/<modelo>/<versión>/` y registra en
`reports/models/s3_publications.json`, por archivo: bucket, key, `VersionId` (si el
bucket tiene versionado), tamaño, SHA-256 local, `ChecksumSHA256` que calcula S3 y el
SHA-256 de una descarga de prueba. El `ETag` se guarda solo como dato de S3: **no** se
usa como SHA-256.

Reglas:
- solo se publica un paquete cuyo sha256 coincide con el registro de OPS-06;
- nunca se sobrescribe una key con otro contenido (una versión publicada es fija);
- la publicación queda registrada solo si head-object, el checksum de S3 y la
  descarga coinciden con el archivo local.

`verify` es el Agent Test: head-object → get-object (del `VersionId` registrado) →
SHA-256 → carga del checkpoint e inferencia de una imagen en **otro proceso** que
solo ve los archivos descargados.

Las credenciales salen de la cadena por defecto de AWS (perfil SSO, variables de
entorno o rol); el código no lee ni guarda access keys.

    uv run python -m classification.publication publish 1.0.0 --bucket <bucket>
    uv run python -m classification.publication verify 1.0.0 --image <crop.png>
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from botocore.exceptions import ClientError

from classification.registry import (
    PACKAGE_CHECKPOINT,
    REGISTRY_PATH,
    REPO_ROOT,
    ModelPackage,
    resolve_checkpoint,
    resolve_model,
)

PUBLICATIONS_PATH = REPO_ROOT / "reports" / "models" / "s3_publications.json"
APP_DIR = REPO_ROOT / "app"
PUBLISHED_FILES = (PACKAGE_CHECKPOINT, "model-card.md", "package.json", "dependencies.json")
CONTENT_TYPES = {
    ".pt": "application/octet-stream",
    ".md": "text/markdown",
    ".json": "application/json",
}
NOT_FOUND = {"404", "NoSuchKey", "NotFound", "NoSuchVersion"}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _checksum_hex(head: dict) -> str | None:
    """`ChecksumSHA256` de S3 (base64) en hexadecimal, si S3 lo devuelve."""
    checksum = head.get("ChecksumSHA256")
    return base64.b64decode(checksum).hex() if checksum else None


def _head(s3, bucket: str, key: str, version_id: str | None = None) -> dict | None:
    kwargs = {"Bucket": bucket, "Key": key, "ChecksumMode": "ENABLED"}
    if version_id:
        kwargs["VersionId"] = version_id
    try:
        return s3.head_object(**kwargs)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in NOT_FOUND:
            return None
        raise


def _download(s3, bucket: str, key: str, version_id: str | None) -> bytes:
    kwargs = {"Bucket": bucket, "Key": key}
    if version_id:
        kwargs["VersionId"] = version_id
    return s3.get_object(**kwargs)["Body"].read()


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["publications"]


def _verified_package(
    version: str, registry_path: Path, repo_root: Path
) -> tuple[ModelPackage, Path]:
    """El paquete local de OPS-06, con cada archivo verificado contra el registro."""
    checkpoint = resolve_checkpoint(version, registry_path=registry_path, repo_root=repo_root)
    package = resolve_model(version, registry_path)
    package_dir = checkpoint.parents[1]
    stored = ModelPackage.model_validate_json((package_dir / "package.json").read_text("utf-8"))
    if stored != package:
        raise ValueError(
            f"package.json de {version} no es la entrada del registro (sha256 distinto)"
        )
    return package, package_dir


def _publish_file(s3, bucket: str, key: str, path: Path, version: str) -> dict:
    body = path.read_bytes()
    sha256 = _sha256(body)
    head = _head(s3, bucket, key)
    if head is not None:
        # Ya publicado: solo vale si es exactamente este archivo.
        if head.get("Metadata", {}).get("sha256") != sha256 or (
            _checksum_hex(head) not in (None, sha256)
        ):
            raise ValueError(f"s3://{bucket}/{key} ya existe con otro contenido: no se sobrescribe")
        version_id = head.get("VersionId")
    else:
        reply = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            Metadata={"sha256": sha256, "model-version": version},
            ChecksumAlgorithm="SHA256",
            ContentType=CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
        )
        version_id = reply.get("VersionId")
        head = _head(s3, bucket, key, version_id)
        if head is None:
            raise ValueError(f"head-object no encuentra s3://{bucket}/{key} recién subido")
    s3_checksum = _checksum_hex(head)
    if s3_checksum is not None and s3_checksum != sha256:
        raise ValueError(
            f"ChecksumSHA256 de S3 para {key} ({s3_checksum}) no es el local ({sha256})"
        )
    if head.get("ContentLength") != len(body):
        raise ValueError(f"{key}: S3 reporta {head.get('ContentLength')} bytes, no {len(body)}")
    downloaded = _sha256(_download(s3, bucket, key, version_id))
    if downloaded != sha256:
        raise ValueError(f"La descarga de prueba de {key} no coincide con el archivo local")
    return {
        "key": key,
        "version_id": version_id,
        "size": len(body),
        "sha256": sha256,
        "s3_checksum_sha256": s3_checksum,
        "download_sha256": downloaded,
        # Dato de S3, no es un SHA-256 (md5 en subidas simples, otro valor en multipart).
        "etag": head.get("ETag", "").strip('"'),
    }


def publish_to_s3(
    version: str,
    *,
    s3,
    bucket: str,
    prefix: str = "models",
    registry_path: Path = REGISTRY_PATH,
    publications_path: Path = PUBLICATIONS_PATH,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Sube y verifica el paquete de `version`; lo registra solo si todo coincide."""
    package, package_dir = _verified_package(version, registry_path, repo_root)
    base = f"{prefix.strip('/')}/{package.model_name}/{package.model_version}"
    files = {
        relpath: _publish_file(s3, bucket, f"{base}/{relpath}", package_dir / relpath, version)
        for relpath in PUBLISHED_FILES
    }
    publications = _load(publications_path)
    for existing in publications:
        if existing["model_version"] == version and existing["bucket"] == bucket:
            if existing["files"] == files:
                return existing
            raise ValueError(f"{version} ya está registrada en s3://{bucket} con otros objetos")
    record = {
        "model_version": str(package.model_version),
        "model_name": package.model_name,
        "run_id": package.run_id,
        "checkpoint_sha256": package.checkpoint_sha256,
        "bucket": bucket,
        "region": getattr(getattr(s3, "meta", None), "region_name", None),
        "published_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    publications.append(record)
    publications_path.parent.mkdir(parents=True, exist_ok=True)
    publications_path.write_text(
        json.dumps({"schema_version": "1.0", "publications": publications}, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


# --- Agent Test ---------------------------------------------------------------------------


def _infer(checkpoint: Path, image: Path) -> dict:
    import torch
    from PIL import Image

    from classification.model import load_checkpoint
    from classification.transforms import eval_transform

    model = load_checkpoint(checkpoint).eval()
    pixels = eval_transform(model.config.image_size)(Image.open(image).convert("RGB"))
    with torch.no_grad():
        scores = torch.softmax(model(pixels.unsqueeze(0)), dim=1)[0].tolist()
    names = {index: name for name, index in model.class_map.items()}
    probabilities = {names[index]: score for index, score in enumerate(scores)}
    return {
        "probabilities": probabilities,
        "predicted_class": max(probabilities, key=probabilities.get),
        "checkpoint_sha256": _sha256(checkpoint.read_bytes()),
        "pid": os.getpid(),
    }


def infer_in_clean_process(checkpoint: Path, image: Path) -> dict:
    """Carga el checkpoint e infiere en un proceso nuevo: solo usa los archivos dados."""
    result = subprocess.run(
        [sys.executable, "-m", "classification.publication", "infer", str(checkpoint), str(image)],
        cwd=APP_DIR,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def verify_publication(
    version: str,
    *,
    s3,
    publications_path: Path = PUBLICATIONS_PATH,
    workdir: Path,
    image: Path,
) -> dict:
    """head-object, get-object, SHA-256, carga e inferencia en un proceso limpio."""
    record = next((p for p in _load(publications_path) if p["model_version"] == version), None)
    if record is None:
        raise KeyError(f"No hay publicación en S3 registrada para {version}")
    bucket = record["bucket"]
    workdir.mkdir(parents=True, exist_ok=True)
    problems, files = [], {}
    for relpath, entry in record["files"].items():
        head = _head(s3, bucket, entry["key"], entry["version_id"])
        check = {
            "key": entry["key"],
            "version_id": entry["version_id"],
            "head_object": head is not None,
            "recorded_sha256": entry["sha256"],
            "download_sha256": None,
            "latest_differs": None,
        }
        files[relpath] = check
        if head is None:
            problems.append(f"{relpath}: head-object no encuentra s3://{bucket}/{entry['key']}")
            continue
        content = _download(s3, bucket, entry["key"], entry["version_id"])
        check["download_sha256"] = _sha256(content)
        if check["download_sha256"] != entry["sha256"]:
            problems.append(f"{relpath}: el SHA-256 descargado no es el registrado")
        target = workdir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        latest = _head(s3, bucket, entry["key"])
        check["latest_differs"] = latest is None or (
            latest.get("VersionId") != entry["version_id"]
            or latest.get("Metadata", {}).get("sha256") != entry["sha256"]
        )
    inference = None
    if not problems:
        inference = infer_in_clean_process(workdir / PACKAGE_CHECKPOINT, image)
        if inference["checkpoint_sha256"] != record["checkpoint_sha256"]:
            problems.append("el proceso limpio cargó otro checkpoint")
    return {
        "model_version": version,
        "bucket": bucket,
        "files": files,
        "inference": inference,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OPS-07: modelo en AWS S3")
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("version")
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--prefix", default="models")
    verify = commands.add_parser("verify")
    verify.add_argument("version")
    verify.add_argument("--image", type=Path, required=True)
    verify.add_argument("--workdir", type=Path)
    infer = commands.add_parser("infer")
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("image", type=Path)
    for sub in (publish, verify):
        sub.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    args = parser.parse_args(argv)

    if args.command == "infer":
        print(json.dumps(_infer(args.checkpoint, args.image)))
        return 0

    import tempfile

    import boto3

    # Cadena de credenciales por defecto (perfil SSO); nunca access keys en el código.
    s3 = boto3.Session(profile_name=args.profile).client("s3")
    if args.command == "publish":
        record = publish_to_s3(args.version, s3=s3, bucket=args.bucket, prefix=args.prefix)
        for entry in record["files"].values():
            print(
                f"s3://{record['bucket']}/{entry['key']} VersionId={entry['version_id']} "
                f"sha256={entry['sha256']} (S3 ChecksumSHA256={entry['s3_checksum_sha256']})"
            )
        return 0
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="ops07-clean-"))
    report = verify_publication(args.version, s3=s3, workdir=workdir, image=args.image)
    for relpath, check in report["files"].items():
        print(
            f"{relpath}: head-object={'OK' if check['head_object'] else 'FALLA'} "
            f"VersionId={check['version_id']} sha256 descargado={check['download_sha256']}"
        )
    if report["inference"]:
        inference = report["inference"]
        print(
            f"proceso limpio (pid {inference['pid']}): {args.image.name} → "
            f"{inference['predicted_class']} {inference['probabilities']}"
        )
    print(f"Descargado en {workdir}")
    print("Publicación verificada" if not report["problems"] else "\n".join(report["problems"]))
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
