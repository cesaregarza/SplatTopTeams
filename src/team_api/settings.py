from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SplatTopTeams API"
    env: str = Field(default="development", alias="ENV")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_rl_per_min: int = Field(default=120, alias="API_RL_PER_MIN")
    rankings_db_schema: str = Field(default="comp_rankings", alias="RANKINGS_DB_SCHEMA")

    def cors_origin_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
