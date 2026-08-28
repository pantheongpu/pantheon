# One-shot repair for the publish workstation after the 2026-08-25 history
# rewrite of pantheongpu_website. Run it from the machine that hosts the
# OneDrive working copies (from WSL:
#   powershell.exe -ExecutionPolicy Bypass -File "$(wslpath -w tools/campaign/fix_publish_workstation.ps1)"
# from the pantheongpu checkout, or run the file directly in PowerShell).
#
# It repoints the clone at the pantheongpu org, discards the stale working
# tree, resets every publish worktree onto the rewritten history, installs
# the hardened campaign script, and verifies no report still carries host
# identifiers. Local uncommitted files in these copies are stale by
# definition and are intentionally discarded.
$ErrorActionPreference = "Stop"

$main = "C:\Users\sssnk\OneDrive\Documents\pantheon"
$docs = "C:\Users\sssnk\OneDrive\Documents"

Write-Host "== Repointing origin at the pantheongpu org"
git -C $main remote set-url origin https://github.com/pantheongpu/pantheongpu_website.git
git -C $main fetch origin

Write-Host "== Resetting the main working copy to origin/main"
git -C $main checkout main
git -C $main reset --hard origin/main

Write-Host "== Resetting publish worktrees onto the rewritten history"
Get-ChildItem $docs -Directory -Filter "pantheon_publish_*" | ForEach-Object {
  Write-Host ("   " + $_.Name)
  git -C $_.FullName checkout --detach origin/main
  git -C $_.FullName reset --hard origin/main
}

Write-Host "== Installing the hardened campaign script"
Copy-Item (Join-Path $PSScriptRoot "lambda_campaign_host.ps1") (Join-Path $main "lambda_campaign_host.ps1") -Force
$staleA100 = Join-Path $main "lambda_a100_campaign.ps1"
if (Test-Path $staleA100) {
  Write-Warning "lambda_a100_campaign.ps1 is hardcoded to a terminated instance; consider deleting it."
}

Write-Host "== Verifying no report carries host identifiers"
$leaks = Get-ChildItem (Join-Path $main "database") -Filter "pantheon_report_*.json" |
  Select-String -Pattern '"network_info"' -List
if ($leaks) {
  Write-Error ("Host identifiers still present in: " + (($leaks | ForEach-Object { Split-Path $_.Path -Leaf }) -join ", "))
}

Write-Host "== Status after repair"
git -C $main status --short | Select-Object -First 5
git -C $main log --oneline -1
Write-Host "Done. Publish workstation is aligned with the rewritten history."
