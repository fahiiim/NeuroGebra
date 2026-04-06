# Pre-Publish Checklist

Use this checklist for every release. It is version-agnostic and should be run from the repository root.

## 1. Version Consistency

- [ ] Update package version in `pyproject.toml`.
- [ ] Update `__version__` in `src/neurogebra/__init__.py`.
- [ ] Add or update release entry in `CHANGELOG.md`.
- [ ] Ensure release notes (if used) match the same version.

## 2. Documentation Consistency

- [ ] `README.md` reflects current API behavior.
- [ ] `docs/index.md` and getting-started docs match current requirements.
- [ ] Remove stale or contradictory claims before release.
- [ ] Confirm framework bridge docs reflect current behavior and limitations.

## 3. Quality Checks

- [ ] Run tests:

```powershell
pytest tests/ -v
```

- [ ] Run formatting/lint/type checks used by your workflow.
- [ ] Build docs:

```powershell
mkdocs build
```

- [ ] Smoke-test key examples:

```powershell
python examples/scripts/basic_usage.py
python examples/scripts/advanced_usage.py
```

## 4. Build Artifacts

- [ ] Clean previous artifacts.
- [ ] Build sdist and wheel.
- [ ] Validate metadata before upload.

```powershell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Get-ChildItem -Filter "*.egg-info" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
```

## 5. Publish

- [ ] (Recommended) Upload to TestPyPI first and verify install.
- [ ] Upload to PyPI.

```powershell
# TestPyPI (optional)
python -m twine upload --repository testpypi dist/*

# PyPI
python -m twine upload dist/*
```

## 6. Release Finalization

- [ ] Create and push git tag for the release.
- [ ] Create GitHub release notes for the tag.
- [ ] Confirm Read the Docs build for the release branch/tag.
- [ ] Verify install in a clean virtual environment.

```powershell
python -m venv .release_check
.\.release_check\Scripts\activate
pip install neurogebra
python -c "import neurogebra; print(neurogebra.__version__)"
deactivate
```

## 7. Post-Release Audit

- [ ] Verify PyPI page renders correctly.
- [ ] Verify README badges and links are healthy.
- [ ] Verify docs deployment is green.
- [ ] Update `PUBLISHING_STATUS.md` with an unambiguous summary.
