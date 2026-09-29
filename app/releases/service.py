import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class P2ReleaseError(Exception):
    """Base error for P2 release selection."""


class P2ReleaseNotFoundError(P2ReleaseError):
    """Raised when the requested P2 release does not exist."""


class P2ReleaseNotApprovedError(P2ReleaseError):
    """Raised when the requested P2 release did not pass the Quality Gate."""


class P2ReleaseUnverifiableError(P2ReleaseError):
    """Raised when a release does not have complete provenance."""


class P2ReleaseService:
    def __init__(self, releases: Mapping[str, Mapping[str, Any]]):
        self._releases = releases

    @classmethod
    def from_catalog(cls, catalog_path: str | Path) -> "P2ReleaseService":
        path = Path(catalog_path)

        if not path.is_file():
            raise P2ReleaseNotFoundError(
                f"P2 release catalog not found: {path}"
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise P2ReleaseUnverifiableError(
                f"Invalid P2 release catalog: {path}"
            ) from exc

        releases_payload = payload.get("releases")

        if not isinstance(releases_payload, list):
            raise P2ReleaseUnverifiableError(
                "P2 release catalog must contain a releases list"
            )

        releases: dict[str, dict[str, Any]] = {}

        for release in releases_payload:
            if not isinstance(release, dict):
                raise P2ReleaseUnverifiableError(
                    "P2 release catalog contains an invalid release"
                )

            release_version = release.get("release_version")

            if not isinstance(release_version, str) or not release_version:
                raise P2ReleaseUnverifiableError(
                    "P2 release is missing release_version"
                )

            releases[release_version] = release

        return cls(releases)

    def select_release(self, release_version: str) -> dict[str, Any]:
        try:
            release = self._releases[release_version]
        except KeyError as exc:
            raise P2ReleaseNotFoundError(
                f"P2 release not found: {release_version}"
            ) from exc

        required_provenance = (
            "dvc_hash",
            "quality_report",
            "coco_path",
            "images_path",
        )

        missing = [
            field
            for field in required_provenance
            if not release.get(field)
        ]

        if missing:
            raise P2ReleaseUnverifiableError(
                f"P2 release has incomplete provenance: "
                f"{release_version}; missing={','.join(missing)}"
            )

        quality_status = release.get("quality_status")

        if quality_status == "failed":
            raise P2ReleaseNotApprovedError(
                f"P2 release is not approved: {release_version}"
            )

        return dict(release)
