# 📋 Pre-Publishing Checklist for Neurogebra v0.2.0

## ✅ Completed Steps

- [✅] Updated version to 0.2.0 in `pyproject.toml`
- [✅] Added scikit-learn to optional dependencies
- [✅] Updated `MANIFEST.in` to include new files
- [✅] Updated `CHANGELOG.md` with v0.2.0 changes
- [✅] Created `RELEASE_NOTES.md`
- [✅] Created `PUBLISHING.md` guide
- [✅] Created `publish.ps1` script
- [✅] Created `assets/` directory for logo
- [✅] Updated `README.md` to reference logo

## ⚠️ Manual Steps Required

### 1. **Save Your Logo Image**
You need to manually save your Neurogebra logo to:
```
c:\Users\WIN\OneDrive\Desktop\Neurogebra\assets\logo.png
```

The logo should be the blue infinity symbol with neural network pattern you showed me.

### 2. **Install Build Tools (if not already installed)**
Open PowerShell and run:
```powershell
pip install --upgrade build twine
```

### 3. **Create PyPI Account (if you don't have one)**
- Go to: https://pypi.org/account/register/
- Verify your email
- Set up 2FA (recommended)

### 4. **Create PyPI API Token**
- Go to: https://pypi.org/manage/account/token/
- Create token with scope "Entire account"
- Save the token securely (starts with `pypi-`)
- You'll need this when publishing

### 5. **Test the Package Locally (Optional but Recommended)**
```powershell
# Clean previous builds
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Get-ChildItem -Filter "*.egg-info" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Build the package
python -m build

# Install locally to test
pip install -e .

# Test imports
python -c "from neurogebra.datasets import Datasets; print(Datasets.list_all())"
python -c "from neurogebra import MathForge; print(MathForge().get('relu'))"
```

## 🚀 Publishing Options

### Option 1: Use the PowerShell Script (Easiest)
```powershell
.\publish.ps1
```

This script will:
- Clean old builds
- Build the package
- Ask for confirmation
- Upload to PyPI (you'll enter credentials)

### Option 2: Manual Publishing
```powershell
# Clean
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Get-ChildItem -Filter "*.egg-info" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Build
python -m build

# Upload
python -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token (the entire `pypi-...` string)

### Option 3: Test on TestPyPI First (Safest)
```powershell
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ neurogebra

# If it works, upload to real PyPI
python -m twine upload dist/*
```

## 📝 After Publishing

### 1. Verify on PyPI
Visit: https://pypi.org/project/neurogebra/

### 2. Test Installation
```powershell
# In a new environment
python -m venv test_env
.\test_env\Scripts\activate
pip install neurogebra
python -c "from neurogebra.datasets import Datasets; print('Success!')"
deactivate
```

### 3. Create Git Tag
```bash
git add .
git commit -m "Release v0.2.0 - Added 100+ datasets"
git tag -a v0.2.0 -m "Release v0.2.0 - Added 100+ datasets"
git push origin main
git push origin v0.2.0
```

### 4. Create GitHub Release
- Go to: https://github.com/fahiiim/NeuroGebra/releases/new
- Select tag: v0.2.0
- Title: "Neurogebra v0.2.0 - 100+ Datasets Update"
- Copy content from `RELEASE_NOTES.md`
- Publish release

### 5. Update README on GitHub
After pushing the logo, update the README image path to:
```markdown
<img src="https://raw.githubusercontent.com/fahiiim/NeuroGebra/main/assets/logo.png" alt="Neurogebra Logo" width="600">
```

## 🐛 Troubleshooting

### "File already exists" error
- You're trying to upload a version that already exists
- You need to increment the version number in `pyproject.toml`
- Clean builds and rebuild: `Remove-Item -Recurse -Force dist, build`

### "Invalid username/password"
- Username must be exactly: `__token__`
- Password is your full API token including `pypi-` prefix
- Make sure there are no extra spaces

### Import errors after install
- Check that all dependencies are in `pyproject.toml`
- Try: `pip install --force-reinstall neurogebra`
- Verify: `pip show neurogebra`

### Package seems incomplete
- Check `MANIFEST.in` includes all necessary files
- Rebuild: `python -m build`
- Inspect the built package: `tar -tf dist/neurogebra-0.2.0.tar.gz`

## 📞 Need Help?

- Check the full guide: `PUBLISHING.md`
- PyPI documentation: https://packaging.python.org/
- Issues: Contact me or file an issue on GitHub

---

## Quick Command Reference

```powershell
# Install tools
pip install --upgrade build twine

# Clean, build, upload (all-in-one)
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue; python -m build; python -m twine upload dist/*

# Or use the script
.\publish.ps1
```
