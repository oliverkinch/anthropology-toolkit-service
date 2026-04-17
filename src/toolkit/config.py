from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    inference_base_url: str = "https://inference.projects.alexandrainst.dk/v1"
    inference_api_key: str = "sk-..."
    default_model: str = "qwen-235b"
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
