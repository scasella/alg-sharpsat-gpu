"""Regression coverage for release documentation link auditing."""

import importlib.util
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    "check_repository", Path(__file__).resolve().parents[1] / "scripts/check_repository.py"
)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.fixture
def guide(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "docs" / "guide.md"
    path.parent.mkdir()
    return path


@pytest.mark.parametrize("prefix", ["", "!"], ids=["link", "image"])
def test_missing_multiline_label_reports_source_line(guide, prefix):
    guide.write_text(
        "# Guide\n\n```md\n[ignored example](missing-example.xml)\n```\n\n"
        f"{prefix}[Recorded verification\nreport](../missing-verification.xml)\n"
    )

    errors, checked = audit.markdown_links(guide)

    assert checked == 1
    assert errors == [
        "docs/guide.md:7: missing local target: ../missing-verification.xml"
    ]


@pytest.mark.parametrize("prefix", ["", "!"], ids=["link", "image"])
def test_valid_multiline_label_is_checked(guide, prefix):
    (guide.parent.parent / "verification.xml").write_text("<testsuites/>\n")
    guide.write_text(f"{prefix}[Recorded verification\nreport](../verification.xml)\n")

    errors, checked = audit.markdown_links(guide)

    assert checked == 1
    assert errors == []
