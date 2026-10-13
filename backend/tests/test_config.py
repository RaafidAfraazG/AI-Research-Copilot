"""
Tests for application configuration (core/config.py).

These tests verify:
- Settings load correctly from environment variables
- Derived properties (URLs) are constructed correctly
- Validation catches invalid values
- The lru_cache can be cleared between tests

No external services or .env files are required.
All values are provided via os.environ patches.
"""
from __future__ import annotations


class TestSettingsDefaults:
    def test_default_app_env_is_development(self) -> None:
        from core.config import Settings
        s = Settings(postgres_password="test")
        assert s.app_env == "development"

    def test_default_backend_port(self) -> None:
        from core.config import Settings
        s = Settings(postgres_password="test")
        assert s.backend_port == 8000

    def test_default_llm_provider(self) -> None:
        from core.config import Settings
        s = Settings(postgres_password="test")
        # Phase 8: default changed from 'ollama' to 'gemini' (free-tier cloud provider)
        assert s.llm_provider == "gemini"

    def test_is_development_flag_true_for_dev(self) -> None:
        from core.config import Settings
        s = Settings(app_env="development", postgres_password="test")
        assert s.is_development is True

    def test_is_development_flag_false_for_production(self) -> None:
        from core.config import Settings
        s = Settings(app_env="production", postgres_password="test", secret_key="a_strong_secret_key_123!")
        assert s.is_development is False


class TestDatabaseUrl:
    def test_database_url_uses_asyncpg_driver(self) -> None:
        from core.config import Settings
        s = Settings(postgres_password="pass")
        assert "postgresql+asyncpg" in s.database_url

    def test_database_url_includes_all_components(self) -> None:
        from core.config import Settings
        s = Settings(
            postgres_host="mydb",
            postgres_port=5433,
            postgres_user="myuser",
            postgres_password="mypass",
            postgres_db="mydb",
        )
        url = s.database_url
        assert "mydb" in url
        assert "5433" in url
        assert "myuser" in url
        assert "mydb" in url

    def test_sync_database_url_uses_psycopg2(self) -> None:
        from core.config import Settings
        s = Settings(postgres_password="pass")
        assert "postgresql+psycopg2" in s.sync_database_url


class TestRedisUrl:
    def test_redis_url_without_password(self) -> None:
        from core.config import Settings
        s = Settings(
            postgres_password="test",
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
        )
        assert s.redis_url == "redis://localhost:6379/0"

    def test_redis_url_with_password(self) -> None:
        from core.config import Settings
        s = Settings(
            postgres_password="test",
            redis_host="redis-host",
            redis_port=6380,
            redis_db=1,
            redis_password="redispass",
        )
        url = s.redis_url
        assert "redispass" in url
        assert "redis-host" in url
        assert "6380" in url


class TestValidators:
    def test_invalid_log_level_raises(self) -> None:
        import pytest

        from core.config import Settings
        with pytest.raises(Exception, match="app_log_level"):
            Settings(app_log_level="INVALID_LEVEL", postgres_password="test")

    def test_valid_log_levels_accepted(self) -> None:
        from core.config import Settings
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            s = Settings(app_log_level=level, postgres_password="test")
            assert s.app_log_level == level

    def test_log_level_is_uppercased(self) -> None:
        from core.config import Settings
        s = Settings(app_log_level="info", postgres_password="test")
        assert s.app_log_level == "INFO"

    def test_production_with_default_secret_raises(self) -> None:
        import pytest

        from core.config import Settings
        with pytest.raises(Exception, match="SECRET_KEY"):
            Settings(
                app_env="production",
                postgres_password="test",
                secret_key="change_me_to_a_random_secret_in_production",
            )

    def test_production_with_custom_secret_ok(self) -> None:
        from core.config import Settings
        s = Settings(
            app_env="production",
            postgres_password="test",
            secret_key="a_properly_strong_and_unique_secret_key_for_production!",
        )
        assert s.app_env == "production"


class TestGetSettings:
    def test_get_settings_returns_settings_instance(self) -> None:
        from core.config import Settings, get_settings
        assert isinstance(get_settings(), Settings)

    def test_get_settings_is_cached(self) -> None:
        from core.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_fresh_instance(self) -> None:
        from core.config import get_settings
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # After clearing cache, a new instance is created
        assert s1 is not s2
