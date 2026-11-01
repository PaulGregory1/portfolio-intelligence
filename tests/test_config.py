from pi.config import Settings


def test_model_slug_is_not_hardcoded() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_model == ""
    assert settings.llm_base_url.startswith("https://")
