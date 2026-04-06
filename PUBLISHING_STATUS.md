# Publishing Status

This file tracks current publishing readiness without conflicting historical notes.

## Current Snapshot

- Package name: `neurogebra`
- Repository version target: `2.5.7` (see `pyproject.toml`)
- Canonical docs config: `.readthedocs.yaml`
- Historical public release evidence exists for `0.2.0`.

## Platform Status

| Platform | Current State | Notes |
|---|---|---|
| GitHub | Active repository | Releases should map to tagged versions in this repo. |
| PyPI | Verify before release | Historical `0.2.0` publication is documented; confirm latest published version before next upload. |
| Read the Docs | Configured | Use `.readthedocs.yaml` as the single source of truth. |

## Pre-Release Gate (Current)

1. Confirm version consistency across:
  - `pyproject.toml`
  - `src/neurogebra/__init__.py`
  - `CHANGELOG.md`
2. Confirm documentation consistency across:
  - `README.md`
  - `docs/index.md`
  - `docs/getting-started/installation.md`
3. Run local quality checks:
  - tests
  - docs build
  - example script smoke checks
4. Build package artifacts and validate metadata:
  - `python -m build`
  - `python -m twine check dist/*`
5. Publish release:
  - upload to PyPI
  - create GitHub tag/release
  - confirm Read the Docs build for the release tag

## Historical Note

The repository contains legacy publishing notes for `v0.2.0` workflows. Those notes are historical context only and should not be used as the current release process.

## Links

- Repository: https://github.com/fahiiim/NeuroGebra
- PyPI: https://pypi.org/project/neurogebra/
- Docs: https://neurogebra.readthedocs.io/
