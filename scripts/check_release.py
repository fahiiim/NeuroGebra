"""Validate package, runtime and optional Git tag versions before publishing."""

import argparse
import re
import sys
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag, for example v3.0.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    version = project["version"]
    runtime = re.search(
        r'^__version__ = "([^"]+)"',
        (root / "src/neurogebra/__init__.py").read_text(encoding="utf-8"),
        re.M,
    )
    if not runtime or runtime[1] != version:
        raise SystemExit("Runtime and package versions differ")
    if args.tag and args.tag != f"v{version}":
        raise SystemExit(f"Tag {args.tag!r} does not match v{version}")
    notes = root / "docs/releases" / f"{version}.md"
    if not notes.is_file() or f"## [{version}]" not in (
        root / "CHANGELOG.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("Missing versioned release notes or changelog entry")
    print(f"Release metadata consistent: {project['name']} {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
