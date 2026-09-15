from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="S10_", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    database_url: SecretStr
    migration_database_url: SecretStr | None = None
    redis_url: SecretStr
    jwt_secret: SecretStr = Field(min_length=32)
    jwt_issuer: str = "s10-platform"
    jwt_audience: str = "s10-api"
    access_minutes: int = Field(default=15, ge=1, le=60)
    refresh_days: int = Field(default=7, ge=1, le=30)
    cors_origins: list[str] = []
    allowed_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    forecast_max_age_days: int = Field(default=10, ge=1, le=30)
    page_size: int = Field(default=20, ge=1, le=100)
    max_page_size: int = Field(default=100, ge=1, le=1000)
    max_offset: int = Field(default=10_000, ge=0, le=100_000)
    query_timeout_ms: int = Field(default=5000, ge=100, le=30_000)
    pool_size: int = Field(default=5, ge=1, le=50)
    global_rate_per_minute: int = Field(default=3000, ge=1)
    ip_rate_per_minute: int = Field(default=120, ge=1)
    login_rate_per_minute: int = Field(default=10, ge=1)
    max_body_bytes: int = Field(default=32_768, ge=1024, le=1_048_576)
    redis_prefix: str = "s10"

    @model_validator(mode="after")
    def validate_configuration(self):
        if make_url(self.database_url.get_secret_value()).get_backend_name() != "postgresql":
            raise ValueError("S10_DATABASE_URL deve usar PostgreSQL")
        if self.page_size > self.max_page_size:
            raise ValueError("S10_PAGE_SIZE excede S10_MAX_PAGE_SIZE")
        if "*" in self.cors_origins or "*" in self.allowed_hosts:
            raise ValueError("Configure origens e hosts explicitos; wildcard nao permitido")
        if self.environment == "production":
            if self.jwt_secret.get_secret_value().startswith("replace-"):
                raise ValueError("Substitua S10_JWT_SECRET por um segredo aleatorio")
            for secret in (self.database_url, self.migration_database_url):
                if secret and make_url(secret.get_secret_value()).query.get("sslmode") not in {
                    "require",
                    "verify-ca",
                    "verify-full",
                }:
                    raise ValueError(
                        "PostgreSQL em producao exige sslmode=require ou verificacao TLS"
                    )
        return self

    def sqlalchemy_url(self, *, migration: bool = False):
        secret = self.migration_database_url if migration else None
        return make_url((secret or self.database_url).get_secret_value()).set(
            drivername="postgresql+psycopg"
        )
