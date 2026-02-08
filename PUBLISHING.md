# Publishing Neurogebra to PyPI

## Prerequisites

1. **Install build tools:**
```bash
pip install --upgrade build twine
```

2. **Create PyPI account:**
   - Go to https://pypi.org/account/register/
   - Verify your email
   - (Optional) Set up 2FA for security

3. **Create API Token:**
   - Go to https://pypi.org/manage/account/token/
   - Create token with scope "Entire account"
   - Save the token securely (starts with `pypi-`)

## Step-by-Step Publishing Process

### 1. Update Version Number
Edit `pyproject.toml` and increment the version:
```toml
version = "0.2.0"  # or whatever the next version should be
```

### 2. Clean Previous Builds
```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info
# On Windows PowerShell:
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
```

### 3. Build the Package
```bash
python -m build
```

This creates:
- `dist/neurogebra-X.X.X.tar.gz` (source distribution)
- `dist/neurogebra-X.X.X-py3-none-any.whl` (wheel distribution)

### 4. Test on TestPyPI (Optional but Recommended)

First upload to TestPyPI:
```bash
python -m twine upload --repository testpypi dist/*
```

Then test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ neurogebra
```

### 5. Upload to PyPI

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your API token (paste the entire `pypi-...` string)

### 6. Verify Upload

Visit: https://pypi.org/project/neurogebra/

Test installation:
```bash
pip install --upgrade neurogebra
```

## Quick Publishing Script

Create `publish.sh` (Linux/Mac) or `publish.ps1` (Windows):

### Windows PowerShell (`publish.ps1`):
```powershell
# Clean build artifacts
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build
python -m build

# Upload
python -m twine upload dist/*

Write-Host "✅ Published successfully!" -ForegroundColor Green
Write-Host "Check: https://pypi.org/project/neurogebra/" -ForegroundColor Cyan
```

Run with:
```bash
.\publish.ps1
```

### Linux/Mac (`publish.sh`):
```bash
#!/bin/bash

# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# Build
python -m build

# Upload
python -m twine upload dist/*

echo "✅ Published successfully!"
echo "Check: https://pypi.org/project/neurogebra/"
```

Run with:
```bash
chmod +x publish.sh
./publish.sh
```

## After Publishing

1. **Create Git Tag:**
```bash
git tag -a v0.2.0 -m "Release v0.2.0 - Added 100+ datasets"
git push origin v0.2.0
```

2. **Create GitHub Release:**
   - Go to: https://github.com/fahiiim/NeuroGebra/releases/new
   - Select the tag you just created
   - Add release notes describing new features
   - Publish release

3. **Update Documentation:**
   - If you have ReadTheDocs, trigger a rebuild
   - Update examples and tutorials as needed

## Troubleshooting

### "File already exists"
- You're trying to upload a version that's already published
- Increment the version number in `pyproject.toml`

### "Invalid username/password"
- Make sure username is exactly `__token__`
- Password should be your full API token including `pypi-` prefix

### "Package name already taken"
- The package name is already registered
- If it's yours, you need the correct credentials
- If it's someone else's, choose a different name

### Import errors after install
- Make sure all dependencies are listed in `pyproject.toml`
- Check that `src/neurogebra/__init__.py` exports are correct
- Try: `pip install --force-reinstall neurogebra`

## Version Numbering Guide

Follow Semantic Versioning (semver.org):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.2.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

For this update (100+ datasets):
- Recommended: `0.2.0` (new feature, backward compatible)

## Checking Package Health

After publishing, verify:
```bash
# Install in clean environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows
pip install neurogebra

# Test imports
python -c "from neurogebra.datasets import Datasets; print(Datasets.list_all())"
python -c "from neurogebra import MathForge; print(MathForge().get('relu'))"

# Deactivate
deactivate
```

## Continuous Integration

Consider setting up GitHub Actions for automated publishing:

Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Store your PyPI token in GitHub Secrets as `PYPI_API_TOKEN`.

---

## Current Package Info

- **Name:** neurogebra
- **Current Version:** Check `pyproject.toml`
- **PyPI URL:** https://pypi.org/project/neurogebra/
- **GitHub:** https://github.com/fahiiim/NeuroGebra
