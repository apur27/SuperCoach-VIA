"""Identity config must resolve in a source checkout and in an installed wheel."""

from pathlib import Path

from supercoach_via.settings import default_config_dir, resolve_config_dir


def test_checkout_config_wins_over_a_packaged_copy(tmp_path: Path) -> None:
    checkout, packaged = tmp_path / "repo", tmp_path / "pkg"
    checkout.mkdir()
    packaged.mkdir()
    (checkout / "team_aliases.csv").write_text("repo\n")
    (packaged / "team_aliases.csv").write_text("pkg\n")
    assert resolve_config_dir(checkout, packaged) == checkout


def test_packaged_config_is_used_when_the_checkout_path_has_no_registry(tmp_path: Path) -> None:
    checkout, packaged = tmp_path / "absent", tmp_path / "pkg"
    packaged.mkdir()
    (packaged / "team_aliases.csv").write_text("pkg\n")
    assert resolve_config_dir(checkout, packaged) == packaged


def test_default_config_dir_has_the_alias_registry() -> None:
    cfg = default_config_dir()
    assert (cfg / "team_aliases.csv").is_file()
    assert (cfg / "venue_aliases.csv").is_file()
    assert (cfg / "coverage.yaml").is_file()


def test_packaged_config_matches_the_checkout_copy() -> None:
    checkout = Path(__file__).resolve().parents[3] / "config"
    packaged = Path(__file__).resolve().parents[3] / "src" / "supercoach_via" / "config"
    names = sorted(p.name for p in checkout.iterdir() if p.is_file())
    assert names
    assert sorted(p.name for p in packaged.iterdir() if p.is_file()) == names
    for name in names:
        assert (packaged / name).read_bytes() == (checkout / name).read_bytes()
    reports = Path(__file__).resolve().parents[3] / "templates" / "reports"
    packaged_reports = Path(__file__).resolve().parents[3] / "src" / "supercoach_via" / "templates" / "reports"
    report_names = sorted(p.name for p in reports.iterdir() if p.is_file())
    assert report_names
    assert sorted(p.name for p in packaged_reports.iterdir() if p.is_file()) == report_names
    for name in report_names:
        assert (packaged_reports / name).read_bytes() == (reports / name).read_bytes()
