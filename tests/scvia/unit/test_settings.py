from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.settings import SettingsError, load_settings


def test_defaults_are_relative_and_portable() -> None:
    s = load_settings(env={})
    assert s.data_root == Path("var") and s.public_base == "/"
    assert "/home/" not in str(s.source_root)


def test_env_overrides_and_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "app.toml"
    cfg.write_text('public_base = "/SuperCoach-VIA/"\ntimezone = "UTC"\n')
    s = load_settings(cfg, env={"SCVIA_DATA_ROOT": str(tmp_path / "d")})
    assert s.public_base == "/SuperCoach-VIA/" and s.data_root == tmp_path / "d"


@pytest.mark.parametrize(
    "content",
    [
        "unknown_key = 1\n",
        'public_base = "no-slash"\n',
        'timezone = "Mars/Base"\n',
        'site_url = "http://x"\n',
        "not toml =",
    ],
)
def test_malformed_settings_fail_loudly(tmp_path: Path, content: str) -> None:
    cfg = tmp_path / "bad.toml"
    cfg.write_text(content)
    with pytest.raises(SettingsError):
        load_settings(cfg, env={})
