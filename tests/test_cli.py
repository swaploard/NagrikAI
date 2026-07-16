from __future__ import annotations

from typer.testing import CliRunner

from nagrik_ai.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "nagrik-ai" in result.output


def test_crawl_help() -> None:
    result = runner.invoke(app, ["crawl", "--help"])
    assert result.exit_code == 0


def test_parse_help() -> None:
    result = runner.invoke(app, ["parse", "--help"])
    assert result.exit_code == 0


def test_vectorize_help() -> None:
    result = runner.invoke(app, ["vectorize", "--help"])
    assert result.exit_code == 0


def test_app_command_help() -> None:
    result = runner.invoke(app, ["app-command", "--help"])
    assert result.exit_code == 0
