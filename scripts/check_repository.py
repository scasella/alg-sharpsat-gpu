#!/usr/bin/env python3
"""Check repository-local Markdown links and publishable artifacts with stdlib.

Run from any directory. The inventory includes tracked and non-ignored new files.
Network links are deliberately excluded: this is a local release audit, not an
external URL availability check. --check-snippets executes the four standalone
Python examples in docs/eli5-and-use-cases.md (offline extraction by default).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SUFFIXES = {".json", ".xml", ".txt", ".md"}
EXCLUDED_PARTS = {
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    "build",
    "dist",
}
SENSITIVE = {
    "machine-specific home path": re.compile(r"/(?:Users|home)/[^\s<>\"']+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"
    ),
    "OpenAI token": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    "AWS access ID": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "credential assignment": re.compile(
        r"(?i)[\"']?(?:api_key|access_token|refresh_token|client_secret|password)"
        r"[\"']?\s*[:=]\s*[\"']([^\"'\s]{12,})[\"']"
    ),
}
INLINE_LINK = re.compile(
    # Labels may wrap across lines, as the verification links in the API guide do.
    r"!?\[[^\]]*\]\(\s*(<[^>\n]+>|(?:\\.|[^()\s]|\([^()]*\))+)"
    r"(?:\s+[\"'][^\n]*?[\"'])?\s*\)"
)
REFERENCE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)", re.M)
REFERENCE_USE = re.compile(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]")


def inventory() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted({ROOT / name for name in result.stdout.decode().split("\0") if name})


def without_fences(text: str) -> str:
    """Preserve line numbers while ignoring backtick/tilde fenced examples."""
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence is None and match:
            fence = match[1]
            lines.append("\n")
        elif fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            lines.append("\n")
        else:
            lines.append(line)
    return "".join(lines)


def anchors(text: str) -> set[str]:
    """GitHub-style heading IDs for this repo's ATX/Setext Markdown headings."""
    text = without_fences(text)
    found = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", text))
    used: set[str] = set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$", line)
        if match:
            title = match[1]
        elif (
            i + 1 < len(lines)
            and line.strip()
            and re.fullmatch(r"\s{0,3}(?:=+|-+)\s*", lines[i + 1])
        ):
            title = line.strip()
        else:
            continue
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title)
        title = html.unescape(re.sub(r"<[^>]*>", "", title)).lower()
        slug = "".join(
            c for c in title if c in "_-" or unicodedata.category(c)[0] not in "PS"
        ).replace(" ", "-")
        unique = slug
        suffix = 0
        while unique in used:
            suffix += 1
            unique = f"{slug}-{suffix}"
        used.add(unique)
        found.add(unique)
    return found


def markdown_links(path: Path) -> tuple[list[str], int]:
    text = without_fences(path.read_text())
    # Ignore inline code examples, while retaining backticks in heading targets.
    text = re.sub(r"(`+)(.*?)\1", lambda m: " " * len(m[0]), text)
    errors = []
    links = [(m.start(), m[1]) for m in INLINE_LINK.finditer(text)]
    references = {m[1].casefold(): m[2] for m in REFERENCE.finditer(text)}
    links.extend((m.start(), m[2]) for m in REFERENCE.finditer(text))
    for match in REFERENCE_USE.finditer(text):
        key = (match[2] or match[1]).casefold()
        if key not in references:
            errors.append(f"{path.relative_to(ROOT)}: undefined reference [{key}]")
    checked = 0
    for offset, raw in links:
        raw = re.sub(r"\\([() ])", r"\1", raw.strip("<>"))
        parsed = urlsplit(raw)
        if parsed.scheme or parsed.netloc:
            continue
        line = text.count("\n", 0, offset) + 1
        prefix = f"{path.relative_to(ROOT)}:{line}"
        target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        checked += 1
        if not target.is_relative_to(ROOT):
            errors.append(f"{prefix}: link leaves repository: {raw}")
        elif not target.exists():
            errors.append(f"{prefix}: missing local target: {raw}")
        elif parsed.fragment and target.suffix.lower() == ".md":
            fragment = unquote(parsed.fragment)
            if fragment not in anchors(target.read_text()):
                errors.append(f"{prefix}: missing Markdown anchor: {raw}")
    return errors, checked


def audit(files: list[Path]) -> tuple[list[str], int]:
    errors = []
    count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if (
            EXCLUDED_PARTS.intersection(relative.parts)
            or any(p.endswith(".egg-info") for p in relative.parts)
            or path.suffix in {".pyc", ".pyo", ".zip"}
        ):
            errors.append(f"{relative}: generated/cache file in publishable inventory")
        if path.suffix.lower() not in ARTIFACT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text()
        for label, pattern in SENSITIVE.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                # Never echo a potential secret into CI logs.
                errors.append(f"{relative}:{line}: possible {label}")
        try:
            if path.suffix == ".json":
                json.loads(text)
            elif path.suffix == ".xml":
                ET.fromstring(text)
        except (ValueError, ET.ParseError) as exc:
            errors.append(f"{relative}: invalid artifact syntax ({type(exc).__name__})")
        if path.suffix.lower() == ".md":
            problems, checked = markdown_links(path)
            errors.extend(problems)
            count += checked
    for directory in (".venv", ".pytest_cache", ".hypothesis", "__pycache__"):
        probe = subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--no-index",
                f"{directory}/release-audit-probe",
            ],
            cwd=ROOT,
        )
        if probe.returncode != 0:
            errors.append(f".gitignore does not ignore {directory}/")
    return errors, count


def check_snippets() -> list[str]:
    path = ROOT / "docs/eli5-and-use-cases.md"
    blocks = re.findall(r"^```python\s*\n(.*?)^```\s*$", path.read_text(), re.M | re.S)
    errors = []
    if not blocks:
        return ["No Python recipes found in docs/eli5-and-use-cases.md"]
    for i, code in enumerate(blocks, 1):
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                errors.append(f"ELI5 recipe {i} failed:\n{result.stderr}")
            else:
                print(f"ELI5 recipe {i}: passed")
        except subprocess.TimeoutExpired:
            errors.append(f"ELI5 recipe {i}: exceeded 60 seconds")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-snippets", action="store_true")
    args = parser.parse_args()
    files = inventory()
    errors, count = audit(files)
    if args.check_snippets:
        errors.extend(check_snippets())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        f"Checked {count} local Markdown links and {len(files)} repository files; no audit findings."
    )
    print(
        "Secret checks cover common token/key patterns; external URLs and Git history are outside this audit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
