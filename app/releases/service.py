import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class P2ReleaseError(Exception):
    """Base error for P2 release selection."""


class P2ReleaseNotFoundError(P2ReleaseError):
    """Raised when the requested P2 release does not exist."""


class P2ReleaseNotApprovedError(P2ReleaseError):
    """Raised when the requested P2 release did not pass the Quality Gate."""


class P2ReleaseUnverifiableError(P2ReleaseError):
    """Raised when a release does not have complete or matching provenance."""


class P2ReleaseService:
    ALLOWED_QUALITY_STATUSES = frozenset({"warning", "passed"})

    def __init__(
        self,
        releases: Mapping[str, Mapping[str, Any]],
        registry: Mapping[str, Mapping[str, Any]] | None = None,
        registry_ref: str | None = None,
    ):
        self._releases = releases
        self._registry = registry
        self._registry_ref = registry_ref

    @staticmethod
    def _resolve_registry_path(
        catalog_path: Path,
        registry_ref: str,
    ) -> Path:
        registry_path = Path(registry_ref)

        if registry_path.is_absolute():
            if registry_path.is_file():
                return registry_path
            raise P2ReleaseUnverifiableError(f"P2 release registry not found: {registry_path}")

        base = catalog_path.resolve().parent

        while True:
            candidate = base / registry_path

            if candidate.is_file():
                return candidate

            if base.parent == base:
                break

            base = base.parent

        raise P2ReleaseUnverifiableError(f"P2 release registry not found: {registry_ref}")

    @classmethod
    def _load_registry(
        cls,
        catalog_path: Path,
        registry_ref: str,
    ) -> dict[str, dict[str, Any]]:
        registry_path = cls._resolve_registry_path(
            catalog_path,
            registry_ref,
        )

        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise P2ReleaseUnverifiableError(
                f"Invalid P2 release registry: {registry_path}"
            ) from exc

        releases_payload = payload.get("releases")

        if not isinstance(releases_payload, list):
            raise P2ReleaseUnverifiableError("P2 release registry must contain a releases list")

        registry: dict[str, dict[str, Any]] = {}

        for release in releases_payload:
            if not isinstance(release, dict):
                raise P2ReleaseUnverifiableError("P2 release registry contains an invalid release")

            dataset_version = release.get("dataset_version")
            quality_file = release.get("quality_file")

            if not isinstance(dataset_version, str) or not dataset_version:
                raise P2ReleaseUnverifiableError("Registered P2 release is missing dataset_version")

            if not isinstance(quality_file, str) or not quality_file:
                raise P2ReleaseUnverifiableError(
                    f"Registered P2 release is missing quality_file: {dataset_version}"
                )

            registry[dataset_version] = release

        return registry

    @classmethod
    def from_catalog(
        cls,
        catalog_path: str | Path,
    ) -> "P2ReleaseService":
        path = Path(catalog_path)

        if not path.is_file():
            raise P2ReleaseNotFoundError(f"P2 release catalog not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise P2ReleaseUnverifiableError(f"Invalid P2 release catalog: {path}") from exc

        releases_payload = payload.get("releases")

        if not isinstance(releases_payload, list):
            raise P2ReleaseUnverifiableError("P2 release catalog must contain a releases list")

        releases: dict[str, dict[str, Any]] = {}

        for release in releases_payload:
            if not isinstance(release, dict):
                raise P2ReleaseUnverifiableError("P2 release catalog contains an invalid release")

            release_version = release.get("release_version")

            if not isinstance(release_version, str) or not release_version:
                raise P2ReleaseUnverifiableError("P2 release is missing release_version")

            releases[release_version] = release

        registry_ref = payload.get("release_registry")
        registry = None

        if registry_ref is not None:
            if not isinstance(registry_ref, str) or not registry_ref:
                raise P2ReleaseUnverifiableError("Invalid release_registry reference")

            registry = cls._load_registry(path, registry_ref)

        service = cls(
            releases,
            registry=registry,
            registry_ref=registry_ref,
        )

        if registry is not None:
            for release_version, release in releases.items():
                service._validate_registry_release(
                    release_version,
                    release,
                )

        return service

    def _validate_registry_release(
        self,
        release_version: str,
        release: Mapping[str, Any],
    ) -> None:
        if self._registry is None or self._registry_ref is None:
            return

        registered = self._registry.get(release_version)

        if registered is None:
            raise P2ReleaseUnverifiableError(
                f"P2 release is not published in official registry: {release_version}"
            )

        quality_file = registered.get("quality_file")

        if not isinstance(quality_file, str) or not quality_file:
            raise P2ReleaseUnverifiableError(
                f"Registered quality_file is invalid: {release_version}"
            )

        registry_parent = PurePosixPath(self._registry_ref).parent
        expected_quality_report = (registry_parent / PurePosixPath(quality_file)).as_posix()

        actual_quality_report = release.get("quality_report")

        if actual_quality_report != expected_quality_report:
            raise P2ReleaseUnverifiableError(
                f"Quality report does not match official registry for {release_version}"
            )

    def select_release(
        self,
        release_version: str,
    ) -> dict[str, Any]:
        try:
            release = self._releases[release_version]
        except KeyError as exc:
            raise P2ReleaseNotFoundError(f"P2 release not found: {release_version}") from exc

        required_provenance = (
            "dvc_hash",
            "quality_report",
            "coco_path",
            "images_path",
        )

        missing = [field for field in required_provenance if not release.get(field)]

        if missing:
            raise P2ReleaseUnverifiableError(
                f"P2 release has incomplete provenance: "
                f"{release_version}; missing={','.join(missing)}"
            )

        self._validate_registry_release(
            release_version,
            release,
        )

        quality_status = release.get("quality_status")

        if quality_status not in self.ALLOWED_QUALITY_STATUSES:
            raise P2ReleaseNotApprovedError(
                f"P2 release has invalid Quality Gate status: "
                f"{release_version}; status={quality_status}"
            )

        return dict(release)

    @staticmethod
    def _read_dvc_hash(path: Path) -> str:
        if not path.is_file():
            raise P2ReleaseUnverifiableError(f"DVC metadata file not found: {path}")

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise P2ReleaseUnverifiableError(f"Invalid DVC metadata file: {path}") from exc

        outs = payload.get("outs") if isinstance(payload, dict) else None

        if not isinstance(outs, list) or not outs:
            raise P2ReleaseUnverifiableError(f"DVC metadata has no outputs: {path}")

        output = outs[0]

        if not isinstance(output, dict):
            raise P2ReleaseUnverifiableError(f"Invalid DVC output metadata: {path}")

        dvc_hash = output.get("md5")

        if not isinstance(dvc_hash, str) or not dvc_hash:
            raise P2ReleaseUnverifiableError(f"DVC metadata has no md5 hash: {path}")

        return dvc_hash

    def verify_release(
        self,
        release_version: str,
        repo_root: str | Path,
    ) -> dict[str, Any]:
        release = self.select_release(release_version)
        root = Path(repo_root)

        verification_fields = (
            "images_dvc_file",
            "images_dvc_hash",
            "annotations_dvc_file",
            "annotations_dvc_hash",
            "quality_report",
        )

        missing = [field for field in verification_fields if not release.get(field)]

        if missing:
            raise P2ReleaseUnverifiableError(
                f"P2 release cannot be verified: {release_version}; missing={','.join(missing)}"
            )

        images_hash = self._read_dvc_hash(root / release["images_dvc_file"])
        annotations_hash = self._read_dvc_hash(root / release["annotations_dvc_file"])

        if images_hash != release["images_dvc_hash"]:
            raise P2ReleaseUnverifiableError(f"Images DVC hash mismatch for {release_version}")

        if annotations_hash != release["annotations_dvc_hash"]:
            raise P2ReleaseUnverifiableError(f"Annotations DVC hash mismatch for {release_version}")

        quality_path = root / release["quality_report"]

        if not quality_path.is_file():
            raise P2ReleaseUnverifiableError(f"Quality report not found: {quality_path}")

        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise P2ReleaseUnverifiableError(f"Invalid quality report: {quality_path}") from exc

        if quality.get("dataset_version") != release_version:
            raise P2ReleaseUnverifiableError(
                f"Quality report version mismatch for {release_version}"
            )

        quality_status = quality.get("status")

        if quality_status != release.get("quality_status"):
            raise P2ReleaseUnverifiableError(f"Quality status mismatch for {release_version}")

        if quality_status not in self.ALLOWED_QUALITY_STATUSES:
            raise P2ReleaseNotApprovedError(
                f"P2 release has invalid Quality Gate status: "
                f"{release_version}; status={quality_status}"
            )

        return dict(release)
