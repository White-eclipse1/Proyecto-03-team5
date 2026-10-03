# OPS-10 — Final Release Audit

Issue: #55 — Run final clean-clone, gates, security and release audit

## Audited revision

- Main commit audited: `c3f14427e85c5ac21d135e8d68ca767a6e09d5f5`
- Audit performed from a separate clean clone.
- Clean clone started with a clean Git working tree.

## M1 — Clean startup

**Status: BLOCKED LOCALLY**

Clean-clone preparation was successful, but the complete local Compose stack could not be started because Docker Desktop could not start its Linux engine. WSL2 reported that virtualization is disabled on the audit machine.

Local evidence:
- Docker CLI and Compose are installed.
- Docker Linux daemon is unavailable.
- Therefore portal, Training page, worker and MLflow were not manually validated from the clean clone.

CI evidence for the audited commit:
- `docker-compose.yml válido`: success.
- `MLflow persistente (OPS-03)`: success.
- `Training worker E2E (OPS-04)`: success.
- `CI OK`: success.

Because the acceptance criteria explicitly require a clean-start validation following the README, M1 is not marked complete from local evidence.

## M2 — Approved P2 release

**Status: AUTOMATED EVIDENCE PASSES**

Targeted release/provenance tests passed as part of the OPS-10 audit.

The repository verifies the registered P2 release `v0.1.1`, including:
- release version;
- DVC hashes for images and annotations;
- quality report association;
- official release registry;
- COCO/images provenance.

The audited test group covering release, provenance, manifests, splits, model registry/publication and evaluation completed with:

- `237 passed`
- `2 skipped`

The two skipped tests require external artifacts/S3 integration and are documented under M4.

GitHub CI additionally reports:
- `Quality gate (fixture)`: success.
- `Quality gate (PROD dataset when OIDC is configured)`: success.

## M3 — No leakage

**Status: AUTOMATED LEAKAGE/SPLIT TESTS PASS; FINAL CRITERIA REVIEW REQUIRED**

The targeted audit includes passing tests for:
- exclusive train/validation/test assignment;
- complete split coverage;
- duplicate groups remaining indivisible;
- rejection of duplicate contamination across splits;
- cross-split near-duplicate leakage detection;
- deterministic stratified split behavior.

The targeted suite passed without failures.

Before marking the issue checkbox as fully complete, the release evidence should explicitly map the final manifest fields required by OPS-10:
- `crop_id` intersection train/val/test = 0;
- `source_image_id` intersection = 0;
- `duplicate_group` intersection = 0;
- augmentation only in train;
- test excluded from candidate selection / early stopping.

## M4 — Reloadable model

**Status: BLOCKED BY IAM/S3 ACCESS**

The repository contains automated tests proving:
- checkpoint publication records SHA-256;
- S3 checksum/download verification;
- model card recovery;
- previous model-version recovery;
- checkpoint loading in a clean process;
- inference from the downloaded checkpoint.

However, the clean-clone real-artifact test was skipped because the required DVC model package and crops were not materialized.

Observed skip:
`Requiere dvc pull data/models.dvc y dvc pull crops`

DVC 3.67.1 was installed successfully and the repository remotes were detected:
- `dev -> s3://dvc-cache`
- `prod -> s3://mlops-p2-dvc-cache-280764207006`

The production DVC pull failed because AWS credentials were unavailable. AWS CLI was then installed and IAM Identity Center configuration attempted, but the user's AWS Identity Center login returned `Access denied`.

Therefore M4 cannot be marked complete until IAM access is restored and the published checkpoint is downloaded and tested from the clean clone.

## Final audit evidence

### Python

Clean clone:

- Python: 3.12
- dependencies installed from locked environment
- full Python test suite: `1091 passed, 26 skipped`
- Ruff: `All checks passed!`

### Backend

Clean clone:

- typecheck: passed
- tests: `104 passed`
- build: passed
- Biome completed with one non-fatal warning in `inference-queue-http.test.ts` regarding a non-null assertion.

### Frontend

Clean clone:

- Biome lint: passed
- typecheck: passed
- production build: passed

On Windows, `reports-plugin.test.ts` cannot create the symlink used by its security fixture and fails with `EPERM`. GitHub CI executes the frontend test job successfully on the audited commit.

### Git / security

- clean-clone working tree: clean
- `git diff --check`: clean
- Ruff: green
- frontend typecheck/build: green
- GitHub `Secret scan (HEAD + Git history)`: success
- GitHub `CI OK`: success
- AWS OIDC verification on main: success

### Dependency audit observations

During clean installation:

Backend npm reported:
- 9 moderate vulnerabilities

Frontend npm reported:
- 12 vulnerabilities
- 5 moderate
- 6 high
- 1 critical

No automatic `npm audit fix --force` was applied during OPS-10 because it can introduce breaking dependency changes. These findings should be reviewed independently before release if the project requires zero dependency advisories.

## GitHub CI evidence

For commit `c3f14427e85c5ac21d135e8d68ca767a6e09d5f5`, all 13 reported GitHub checks completed successfully, including:

- CI OK
- Python (Ruff y pytest)
- Backend (Biome, tsc, tests y build)
- Frontend (Biome, tsc, tests y build)
- docker-compose.yml válido
- Quality gate fixture
- Quality gate PROD
- MLflow persistente
- Training worker E2E
- Secret scan HEAD + Git history
- AWS OIDC verification
- Terraform validation

## Remaining blockers

OPS-10 is not ready to close yet because:

1. M1 local clean-start validation is blocked by Docker/WSL2 virtualization on the audit workstation.
2. M4 real artifact recovery is blocked by IAM Identity Center / S3 access.
3. APP-10 (#52), an explicit dependency of OPS-10, is still open.
4. Final release/tag must not be created until all required audit criteria pass.

## Release decision

**Release candidate is not tagged from OPS-10 yet.**

The repository-level automated gates are green, but external/environmental blockers prevent completing every OPS-10 acceptance criterion at this time.
