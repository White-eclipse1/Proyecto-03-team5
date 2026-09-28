import pytest
from pydantic import ValidationError

from storage.settings import Settings

REQUIRED_ENV = {
    "DATABASE_URL": "mysql+pymysql://user:pass@localhost/db",
    "MINIO_ENDPOINT": "localhost",
    "MINIO_PORT": "9000",
    "MINIO_ACCESS_KEY": "ak",
    "MINIO_SECRET_KEY": "sk",
    "MINIO_BUCKET": "bucket",
    "DATASET_DIR": "/data/raw",
    "REPORTS_DIR": "/reports",
}


def set_required_env(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


def test_settings_load_env_and_real_yaml_files(monkeypatch):
    set_required_env(monkeypatch)
    settings = Settings()
    assert settings.database_url == REQUIRED_ENV["DATABASE_URL"]
    assert settings.minio_port == 9000
    assert settings.minio_use_ssl is False
    assert settings.quality.min_images_per_class.threshold == 300


@pytest.mark.parametrize("missing", list(REQUIRED_ENV))
def test_missing_env_var_is_rejected_naming_the_field_not_a_key_error(monkeypatch, missing):
    set_required_env(monkeypatch)
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ValidationError, match=missing.lower()):
        Settings()


def test_non_numeric_port_is_rejected(monkeypatch):
    set_required_env(monkeypatch)
    monkeypatch.setenv("MINIO_PORT", "not-a-port")
    with pytest.raises(ValidationError, match="minio_port"):
        Settings()


@pytest.mark.parametrize("override", ['{"min_images_per_class": {"threshold": 999}}', "not-json"])
def test_quality_env_cannot_override_managed_yaml(monkeypatch, override):
    set_required_env(monkeypatch)
    monkeypatch.setenv("QUALITY", override)
    assert Settings().quality.min_images_per_class.threshold == 300


def test_quality_dotenv_cannot_override_managed_yaml(monkeypatch, tmp_path):
    set_required_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("QUALITY=not-json\nDATASET_VERSION=dotenv-test\n", encoding="utf-8")
    monkeypatch.delenv("DATASET_VERSION", raising=False)
    settings = Settings(_env_file=env_file)
    assert settings.quality.min_images_per_class.threshold == 300
    assert settings.dataset_version == "dotenv-test"


def test_next_settings_load_observes_managed_yaml(monkeypatch, tmp_path):
    import yaml

    import storage.settings as module
    from policies.models import load_quality_policy
    from splits.models import load_splits_config

    set_required_env(monkeypatch)
    policy = load_quality_policy()
    directory = tmp_path / "policies"
    directory.mkdir()
    path = directory / "quality.yaml"
    path.write_text(yaml.safe_dump(policy.model_dump()), encoding="utf-8")
    monkeypatch.setattr(module, "APP_ROOT", tmp_path)
    before = Settings()
    policy.min_images_per_class.threshold = 77.0
    path.write_text(yaml.safe_dump(policy.model_dump()), encoding="utf-8")
    after = Settings()
    assert before.quality.min_images_per_class.threshold == 300
    assert after.quality.min_images_per_class.threshold == 77
    assert load_quality_policy(path) == after.quality
    split_path = tmp_path / "splits.yaml"
    split_path.write_text("train: 0.5\nval: 0.3\ntest: 0.2\nseed: 71\n")
    config = load_splits_config(split_path)
    assert (config.train, config.val, config.test, config.seed) == (0.5, 0.3, 0.2, 71)


def test_copilot_settings_are_optional_with_a_safe_default(monkeypatch):
    set_required_env(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("COPILOT_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is None
    assert settings.copilot_model == "claude-opus-5"
    assert settings.copilot_host == "127.0.0.1"
    assert settings.copilot_port == 8000


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_anthropic_api_key_means_not_configured(monkeypatch, blank):
    # docker-compose pasa `${ANTHROPIC_API_KEY:-}` como cadena vacía si .env no la define.
    set_required_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", blank)
    assert Settings(_env_file=None).anthropic_api_key is None


def test_anthropic_api_key_is_read_from_env_and_never_leaks_in_repr(monkeypatch):
    set_required_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret")
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-secret"
    assert "sk-ant-test-secret" not in repr(settings)
    assert "sk-ant-test-secret" not in str(settings)
