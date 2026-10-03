"""APP-09 (revisión del PR #63): ml-api confirma en S3 lo que Models marca `published`.

`training.s3_verification.BotoS3Verifier` hace `head_object` con `ChecksumMode=ENABLED`
(y el `VersionId` registrado) y devuelve el `ChecksumSHA256` en hexadecimal. Un objeto
que no existe es `None`; si S3 no se puede consultar (sin credenciales, sin permiso,
sin red) lanza `S3Unverifiable`, nunca otra excepción. Las respuestas se guardan unos
segundos por (bucket, key, version_id) para no consultar S3 en cada petición.
"""

import base64
import hashlib

import boto3
import pytest
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
from botocore.stub import Stubber

from training.s3_verification import CACHE_SECONDS, BotoS3Verifier, S3Head, S3Unverifiable

BUCKET = "mlops-p2-dvc-cache-280764207006"
KEY = "models/dog-cat-resnet18/1.0.0/model-card.md"
DIGEST = hashlib.sha256(b"# model card\n").digest()


def stubbed_client():
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    return client, Stubber(client)


def expect_head(stubber, *, version_id=None, checksum=DIGEST):
    params = {"Bucket": BUCKET, "Key": KEY, "ChecksumMode": "ENABLED"}
    if version_id:
        params["VersionId"] = version_id
    response = {"ContentLength": 13}
    if checksum is not None:
        response["ChecksumSHA256"] = base64.b64encode(checksum).decode()
    stubber.add_response("head_object", response, params)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def verifier_for(client, clock=None) -> BotoS3Verifier:
    return BotoS3Verifier(client_for_region=lambda _region: client, clock=clock or Clock())


def head(verifier, version_id=None):
    return verifier.head(bucket=BUCKET, region="us-east-1", key=KEY, version_id=version_id)


def test_existing_object_returns_its_checksum_in_hex():
    client, stubber = stubbed_client()
    expect_head(stubber)

    with stubber:
        assert head(verifier_for(client)) == S3Head(checksum_sha256=DIGEST.hex())
        stubber.assert_no_pending_responses()


def test_the_registered_version_id_is_the_one_asked_for():
    client, stubber = stubbed_client()
    expect_head(stubber, version_id="3HL4kqtJlcpXroDTDmJ.rmSpXd3dIbrHY")

    with stubber:
        result = head(verifier_for(client), version_id="3HL4kqtJlcpXroDTDmJ.rmSpXd3dIbrHY")

    assert result == S3Head(checksum_sha256=DIGEST.hex())


def test_object_without_checksum_has_no_checksum():
    client, stubber = stubbed_client()
    expect_head(stubber, checksum=None)

    with stubber:
        assert head(verifier_for(client)) == S3Head(checksum_sha256=None)


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NoSuchVersion"])
def test_missing_object_is_none(code):
    client, stubber = stubbed_client()
    stubber.add_client_error("head_object", service_error_code=code, http_status_code=404)

    with stubber:
        assert head(verifier_for(client)) is None


@pytest.mark.parametrize(
    ("code", "status"),
    [("403", 403), ("AccessDenied", 403), ("ExpiredToken", 400), ("InternalError", 500)],
)
def test_s3_errors_that_are_not_a_missing_object_are_unverifiable(code, status):
    client, stubber = stubbed_client()
    stubber.add_client_error("head_object", service_error_code=code, http_status_code=status)

    with stubber, pytest.raises(S3Unverifiable) as raised:
        head(verifier_for(client))

    assert KEY not in str(raised.value)


class Failing:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def head_object(self, **_kwargs):
        self.calls += 1
        raise self.error


@pytest.mark.parametrize(
    "error",
    [NoCredentialsError(), EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")],
    ids=["no-credentials", "no-network"],
)
def test_without_credentials_or_network_it_is_unverifiable(error):
    with pytest.raises(S3Unverifiable) as raised:
        head(verifier_for(Failing(error)))

    assert str(raised.value)
    assert KEY not in str(raised.value)


def test_without_credentials_the_message_says_so():
    with pytest.raises(S3Unverifiable, match="credenciales"):
        head(verifier_for(Failing(NoCredentialsError())))


def test_a_client_that_cannot_be_built_is_unverifiable():
    def broken(_region):
        raise NoCredentialsError()

    with pytest.raises(S3Unverifiable):
        BotoS3Verifier(client_for_region=broken).head(
            bucket=BUCKET, region="us-east-1", key=KEY, version_id=None
        )


def test_answers_are_cached_for_a_while_per_object():
    client, stubber = stubbed_client()
    clock = Clock()
    verifier = verifier_for(client, clock)
    expect_head(stubber)
    expect_head(stubber, version_id="v2")
    expect_head(stubber, checksum=hashlib.sha256(b"otra").digest())

    with stubber:
        first = head(verifier)
        clock.now += CACHE_SECONDS - 1
        assert head(verifier) == first, "dentro del TTL no se vuelve a consultar S3"
        assert head(verifier, version_id="v2") == first, "otra versión es otro objeto"
        clock.now += 2
        assert head(verifier) == S3Head(checksum_sha256=hashlib.sha256(b"otra").hexdigest())
        stubber.assert_no_pending_responses()


def test_unverifiable_answers_are_cached_too():
    failing = Failing(NoCredentialsError())
    clock = Clock()
    verifier = verifier_for(failing, clock)

    for _ in range(3):
        with pytest.raises(S3Unverifiable):
            head(verifier)
    assert failing.calls == 1
    clock.now += CACHE_SECONDS + 1
    with pytest.raises(S3Unverifiable):
        head(verifier)
    assert failing.calls == 2


def test_the_client_is_built_for_the_region_of_the_publication():
    regions = []
    client, stubber = stubbed_client()
    expect_head(stubber)

    def factory(region):
        regions.append(region)
        return client

    with stubber:
        BotoS3Verifier(client_for_region=factory).head(
            bucket=BUCKET, region="us-east-1", key=KEY, version_id=None
        )

    assert regions == ["us-east-1"]
