# Publish NeuroGebra 3.0.0

Publishing is a maintainer action. Preparing the working tree or pushing a branch/tag alone does not upload to PyPI. The **Publish to PyPI** workflow runs when a GitHub release is published.

## Validate locally

Use Python 3.11+ for the release metadata checker and Node.js 24 for the website interaction checks. Create an isolated environment and install the tools:

```bash
python -m pip install -e ".[dev,docs]" build twine
python scripts/check_release.py --tag v3.0.0
python -m pytest tests -q
python -m mkdocs build --strict
python scripts/check_docs.py --site site
node --test tests/web/home-demo.test.cjs
python -m build --outdir dist/3.0.0
python -m twine check dist/3.0.0/*
```

`publish.ps1` runs the local checks and prepares version-specific artifacts; it performs no push or upload. Keep historical artifacts outside the release upload directory.

## Push and create the release

Review the complete diff and include all intended new files. Then run these commands yourself from your chosen release branch:

```bash
git add src tests docs overrides scripts .github .readthedocs.yaml pyproject.toml mkdocs.yml README.md CHANGELOG.md RELEASE_NOTES.md PUBLISHING.md PUBLISHING_STATUS.md publish.ps1
git commit -m "Release NeuroGebra 3.0.0"
git push origin HEAD
git tag -a v3.0.0 -m "NeuroGebra 3.0.0"
git push origin v3.0.0
```

In GitHub, open **Releases → Draft a new release**. Select `v3.0.0`, use the title **NeuroGebra 3.0.0**, paste `RELEASE_NOTES.md`, and publish it as a stable release. Set it as the latest release so the repository sidebar updates from the older version.

Before publishing, ensure the repository Actions secret `PYPI_TOKEN` contains a token authorized for the `neurogebra` project. The workflow retains the existing token-based setup; no secret is included in this checkout. A GitHub prerelease uploads to TestPyPI using `TEST_PYPI_TOKEN` instead.

## What the workflow checks

The workflow checks out the release tag, validates version agreement, runs the core tests, builds strict documentation, checks snippets and local links, builds distributions, validates package metadata and imports the installed wheel outside the checkout. Only the resulting checked artifacts reach the upload job. Duplicate uploads fail visibly.

The regular cross-platform test matrix remains a separate workflow; wait for it to pass on the release commit too. Optional framework and hardware tests need their own supported environments.

## Verify publication

1. Confirm the **Publish to PyPI** workflow succeeds for `v3.0.0`.
2. Confirm the [GitHub release](https://github.com/fahiiim/NeuroGebra/releases) is marked latest and [PyPI](https://pypi.org/project/neurogebra/) lists 3.0.0.
3. In a fresh environment, run `python -m pip install "neurogebra==3.0.0"` and inspect `neurogebra.__version__`.
4. Confirm the GitHub Pages documentation workflow and the Read the Docs build succeed. Read the Docs uses `.readthedocs.yaml`; verify its webhook/version activation if no build starts.
5. Update `PUBLISHING_STATUS.md` with verified results.

If an upload fails, inspect the failed check before retrying. PyPI versions are immutable; a published artifact cannot be replaced under the same version.
