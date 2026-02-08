# Neurogebra Publishing Script for Windows PowerShell

Write-Host "`n🚀 Neurogebra Publishing Script" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Step 1: Clean previous builds
Write-Host "📦 Step 1: Cleaning previous builds..." -ForegroundColor Yellow
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Get-ChildItem -Filter "*.egg-info" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cleaned!`n" -ForegroundColor Green

# Step 2: Run tests (optional - uncomment if you want)
# Write-Host "🧪 Step 2: Running tests..." -ForegroundColor Yellow
# python -m pytest tests/ -v
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "❌ Tests failed! Aborting." -ForegroundColor Red
#     exit 1
# }
# Write-Host "✅ Tests passed!`n" -ForegroundColor Green

# Step 3: Build the package
Write-Host "🔨 Step 2: Building package..." -ForegroundColor Yellow
python -m build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed! Check errors above." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build successful!`n" -ForegroundColor Green

# Step 4: Show what was built
Write-Host "📋 Built files:" -ForegroundColor Cyan
Get-ChildItem dist/ | ForEach-Object { Write-Host "   - $($_.Name)" -ForegroundColor White }
Write-Host ""

# Step 5: Ask for confirmation
Write-Host "⚠️  Ready to upload to PyPI" -ForegroundColor Yellow
Write-Host "   Make sure you've updated the version in pyproject.toml!" -ForegroundColor Yellow
$confirmation = Read-Host "`n   Continue? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Host "`n❌ Upload cancelled." -ForegroundColor Red
    exit 0
}

# Step 6: Upload to PyPI
Write-Host "`n📤 Step 3: Uploading to PyPI..." -ForegroundColor Yellow
Write-Host "   (You'll need to enter your PyPI credentials)" -ForegroundColor Gray
python -m twine upload dist/*

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Upload failed! Check errors above." -ForegroundColor Red
    exit 1
}

# Success!
Write-Host "`n" -NoNewline
Write-Host "🎉 SUCCESS! Package published to PyPI!" -ForegroundColor Green
Write-Host "================================`n" -ForegroundColor Cyan

Write-Host "📦 Check your package at:" -ForegroundColor Cyan
Write-Host "   https://pypi.org/project/neurogebra/`n" -ForegroundColor White

Write-Host "💡 Test installation with:" -ForegroundColor Cyan
Write-Host "   pip install --upgrade neurogebra`n" -ForegroundColor White

Write-Host "🏷️  Don't forget to:" -ForegroundColor Yellow
Write-Host "   1. Create a git tag: git tag -a vX.X.X -m 'Release vX.X.X'" -ForegroundColor White
Write-Host "   2. Push the tag: git push origin vX.X.X" -ForegroundColor White
Write-Host "   3. Create a GitHub release`n" -ForegroundColor White
