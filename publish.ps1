# Prepare and validate a release locally. Publication is performed by the maintainer.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $releaseVersion = '3.0.0'
    $checks = @(
        @('scripts/check_release.py', '--tag', "v$releaseVersion"),
        @('-m', 'pytest', 'tests', '-q'),
        @('-m', 'mkdocs', 'build', '--strict'),
        @('scripts/check_docs.py', '--site', 'site'),
        @('-m', 'build', '--outdir', "dist/$releaseVersion")
    )
    foreach ($check in $checks) {
        & python @check
        if ($LASTEXITCODE -ne 0) { throw "Release check failed: python $check" }
    }
    $artifacts = @(Get-ChildItem -LiteralPath "dist/$releaseVersion" -File | ForEach-Object { $_.FullName })
    & python -m twine check @artifacts
    if ($LASTEXITCODE -ne 0) { throw 'Distribution metadata validation failed' }
    Write-Host "Release $releaseVersion prepared. Follow PUBLISHING.md to push and publish."
} finally {
    Pop-Location
}
