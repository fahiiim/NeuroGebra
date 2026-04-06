# ReadTheDocs Setup Guide

This guide documents the current Read the Docs setup for this repository.

## Canonical Configuration

Use only:

- `.readthedocs.yaml`

Do not maintain a parallel `.readthedocs.yml` file. A single config avoids ambiguous builds.

## Current Build Profile

The canonical config currently specifies:

- Config version: 2
- OS: Ubuntu 22.04
- Python: 3.11
- Docs builder: MkDocs (`mkdocs.yml`)
- Python install method: pip with `docs` extra

## Initial Setup

1. Go to https://readthedocs.org/
2. Sign in with GitHub.
3. Import repository `fahiiim/NeuroGebra`.
4. Confirm the project uses `.readthedocs.yaml`.
5. Trigger first build and verify status.

Expected docs URL:

- https://neurogebra.readthedocs.io/

## Local Validation

Before pushing docs changes, validate locally:

```powershell
pip install -e .[docs]
mkdocs build
mkdocs serve
```

## Recommended Project Settings (Read the Docs)

- Default branch: `main`
- Enable pull request builds
- Keep `latest` and `stable` versions active

## Troubleshooting

### Build fails with import errors

- Verify `pyproject.toml` includes required docs dependencies.
- Verify `.readthedocs.yaml` installs package with docs extras.

### MkDocs errors

- Validate `mkdocs.yml` syntax.
- Re-run `mkdocs build` locally to reproduce.

### Docs do not update

- Confirm GitHub webhook exists and is active.
- Trigger a manual build from Read the Docs dashboard.

## Useful Links

- Project dashboard: https://readthedocs.org/projects/neurogebra/
- Build history: https://readthedocs.org/projects/neurogebra/builds/
- RTD docs: https://docs.readthedocs.io/
