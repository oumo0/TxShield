from pathlib import Path

import pytest

from txshield.config.yaml_loader import cli_or_config, load_yaml_mapping


def test_load_yaml_mapping(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("paths:\n  input: data/raw\n", encoding="utf-8")
    assert load_yaml_mapping(config_file)["paths"]["input"] == "data/raw"


def test_cli_value_has_precedence():
    assert cli_or_config("cli", {"value": "yaml"}, "value") == "cli"
    assert cli_or_config(None, {"value": "yaml"}, "value") == "yaml"


def test_yaml_root_must_be_mapping(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- invalid\n- root\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_yaml_mapping(config_file)
