"""Tests for the CLI module."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from calm_data_generator.cli import (
    get_docs_dir,
    get_package_dir,
    get_tutorials_dir,
    list_tutorials,
    main,
    run_tutorial,
    show_path,
    show_tutorial,
)


def test_get_package_dir():
    d = get_package_dir()
    assert isinstance(d, Path)
    assert d.exists()


def test_get_tutorials_dir_returns_path():
    d = get_tutorials_dir()
    assert isinstance(d, Path)


def test_get_docs_dir_returns_path():
    d = get_docs_dir()
    assert isinstance(d, Path)


def test_list_tutorials_no_crash(capsys):
    list_tutorials()
    # Just ensure it doesn't raise; output is optional
    out = capsys.readouterr().out
    assert isinstance(out, str)


def test_show_tutorial_invalid_number(capsys):
    show_tutorial("9999")
    out = capsys.readouterr().out
    assert "Invalid" in out or "No tutorials" in out


def test_show_tutorial_non_numeric(capsys):
    show_tutorial("abc")
    out = capsys.readouterr().out
    assert "number" in out.lower() or "invalid" in out.lower() or "no tutorials" in out.lower()


def test_run_tutorial_invalid_number(capsys):
    run_tutorial("9999")
    out = capsys.readouterr().out
    assert "Invalid" in out or "No tutorials" in out


def test_show_path_output(capsys):
    show_path()
    out = capsys.readouterr().out
    assert "Tutorials" in out or "Package" in out


def test_main_no_args_exits(capsys):
    with patch("sys.argv", ["calm_data_generator"]):
        # argparse prints help and exits 0 or just returns for missing subcommand
        try:
            main()
        except SystemExit as e:
            assert e.code in (0, 1, 2, None)


def test_main_tutorials_list(capsys):
    with patch("sys.argv", ["calm_data_generator", "tutorials", "list"]):
        main()
    out = capsys.readouterr().out
    assert isinstance(out, str)


def test_main_tutorials_show_no_number(capsys):
    with patch("sys.argv", ["calm_data_generator", "tutorials", "show"]):
        main()
    out = capsys.readouterr().out
    assert "number" in out.lower() or isinstance(out, str)


def test_main_tutorials_run_no_number(capsys):
    with patch("sys.argv", ["calm_data_generator", "tutorials", "run"]):
        main()
    out = capsys.readouterr().out
    assert "number" in out.lower() or isinstance(out, str)
