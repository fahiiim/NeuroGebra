# ReadTheDocs Setup Guide for Neurogebra

## Quick Start

Your documentation is ready to be hosted on ReadTheDocs! Here's how to set it up:

### 1. Import Your Project

1. Go to **https://readthedocs.org/**
2. Click **"Sign Up"** or **"Log In"** with your GitHub account
3. Click **"Import a Project"**
4. Find **"NeuroGebra"** in the list and click **"+"**
5. Click **"Next"** (default settings are fine)

### 2. Verify Configuration

ReadTheDocs will automatically use the `.readthedocs.yaml` file in your repository. This file configures:
- Python 3.11
- Ubuntu 22.04
- MkDocs for documentation building
- Automatic installation of docs dependencies

### 3. Build Documentation

1. ReadTheDocs will automatically trigger a build after import
2. Wait for the build to complete (usually 2-5 minutes)
3. Visit your documentation at: **https://neurogebra.readthedocs.io/**

### 4. Enable Webhooks (Automatic Builds)

ReadTheDocs should automatically set up a webhook in your GitHub repository. This means:
- Every push to `main` triggers a new documentation build
- Pull requests get preview documentation builds
- Tags/releases get versioned documentation

To verify webhooks:
1. Go to your GitHub repo: https://github.com/fahiiim/NeuroGebra
2. Click **Settings** → **Webhooks**
3. You should see a webhook for `readthedocs.org`

### 5. Configure Build Settings (Optional)

In ReadTheDocs project settings, you can:

**Enable/Disable Builds:**
- `Admin` → `Advanced Settings`
- Set **"Default branch"** to `main`
- Enable **"Build pull requests"** for PR previews

**Versioning:**
- `Versions` tab
- Activate versions you want (e.g., `latest`, `stable`, `v0.2.0`)

**Notifications:**
- `Notifications` → Add email for build failure alerts

### 6. Custom Domain (Optional)

To use a custom domain like `docs.neurogebra.com`:

1. In ReadTheDocs: `Admin` → `Domains` → Add domain
2. In your DNS provider: Add CNAME record pointing to `neurogebra.readthedocs.io`
3. Wait for DNS propagation (can take up to 48 hours)

## Documentation Structure

Your project uses **MkDocs** with the following structure:

```
docs/
├── index.md              # Homepage
├── getting-started.md    # Installation & Quick Start
├── api/
│   └── reference.md     # API Documentation
├── examples/
│   ├── custom_activation.md
│   ├── custom_loss.md
│   └── training_expressions.md
└── tutorials/
    ├── beginner.md
    ├── intermediate.md
    └── advanced.md
```

**Configuration:** `mkdocs.yml`

## Debugging Build Failures

If your build fails:

1. **Check Build Logs:**
   - Click on the failed build in ReadTheDocs
   - Review the error messages

2. **Common Issues:**
   - Missing dependencies: Update `pyproject.toml` `[project.optional-dependencies.docs]`
   - Invalid Markdown: Check for syntax errors in `.md` files
   - MkDocs config errors: Validate `mkdocs.yml`

3. **Test Locally:**
   ```powershell
   # Install docs dependencies
   pip install -e .[docs]
   
   # Build docs locally
   mkdocs build
   
   # Serve docs locally (http://127.0.0.1:8000)
   mkdocs serve
   ```

## Advanced Features

### Search

ReadTheDocs provides built-in search across all your documentation automatically.

### Versioned Documentation

ReadTheDocs creates separate documentation for each tag/release:
- `latest` - built from `main` branch
- `stable` - latest release tag
- `v0.2.0` - specific version

### Download Formats

ReadTheDocs can build PDF, EPUB, and HTMLZip formats. Enable in:
`Admin` → `Advanced Settings` → Check desired formats

### Analytics

Enable traffic analytics:
`Admin` → `Advanced Settings` → Enable **"Analytics"**

## Updating Documentation

After pushing changes to GitHub:

1. **Automatic:** ReadTheDocs webhook triggers a build automatically
2. **Manual:** Click **"Build Version"** button in ReadTheDocs dashboard

## Status Badge

Add a ReadTheDocs status badge to your README.md:

```markdown
[![Documentation Status](https://readthedocs.org/projects/neurogebra/badge/?version=latest)](https://neurogebra.readthedocs.io/en/latest/?badge=latest)
```

## Links

- **Your Documentation:** https://neurogebra.readthedocs.io/
- **Project Dashboard:** https://readthedocs.org/projects/neurogebra/
- **Build History:** https://readthedocs.org/projects/neurogebra/builds/
- **ReadTheDocs Docs:** https://docs.readthedocs.io/

## Troubleshooting

### "Project not found"
- Make sure you've imported the project on ReadTheDocs
- Check that your GitHub repository is public or you've granted access

### "Build failed: No module named 'neurogebra'"
- The `.readthedocs.yaml` should install your package with `pip install -e .[docs]`
- Check that `pyproject.toml` has the correct `docs` extra dependencies

### "MkDocs configuration error"
- Validate `mkdocs.yml` syntax
- Test locally with `mkdocs build`

### Documentation not updating
- Check webhook is active in GitHub settings
- Manually trigger a build in ReadTheDocs
- Check build logs for errors

---

## Summary

✅ Configuration file created: `.readthedocs.yaml`
✅ Ready to import on https://readthedocs.org/
✅ Automatic builds on every git push
✅ Documentation will be live at: https://neurogebra.readthedocs.io/

**Next steps:**
1. Import project on ReadTheDocs
2. Wait for first build to complete
3. Share your documentation link!
