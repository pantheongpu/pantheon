# Runs the full Pantheon workload queue on a rented Lambda instance and
# publishes each finished report to the public website repository.
#
# Source of truth lives in the private pantheongpu repo (tools/campaign/).
# Install the current copy onto the publish workstation with
# fix_publish_workstation.ps1 from this directory.
param(
  [Parameter(Mandatory=$true)][string]$Ip,
  [Parameter(Mandatory=$true)][string]$GpuSelector,
  [Parameter(Mandatory=$true)][string]$Label,
  [Parameter(Mandatory=$true)][string]$Website,
  [Parameter(Mandatory=$true)][string]$InstanceId
)
$ErrorActionPreference = "Continue"
$remote = "ubuntu@$Ip"
$key = "/home/saqib/.ssh/id_rsa"
$ssh = "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i $key $remote"
$scp = "scp -o StrictHostKeyChecking=no -o BatchMode=yes -i $key"
$workloads = @(
  "memory_thermal_asym", "memory_cache_fracture", "memory_retention_bake", "memory_pc_pingpong",
  "memory_bank_thrash", "memory_tsv_thrasher", "tlb_avalanche", "ras_validator", "int_virus",
  "scheduler", "mma_virus", "fp64_virus", "memory_write", "memory_write_agg", "memory_read",
  "memory_read_agg", "voltage", "incinerator", "cache_lat", "sfu_stress", "pcie_bandwidth",
  "pulse_virus", "tensor_virus", "atomic_virus", "omni_virus", "p2p_thrasher", "all_reduce",
  "transformer_virus", "llm_decode", "llm_prefill", "kv_cache_churn", "fused_attention", "rope_stress",
  "quantized_gemm", "serving_mix", "speculative_decode", "moe_router", "transformer_train_step",
  "allocation_fragmentation", "graph_replay", "rag_embedding", "vision_encoder", "rt_virus", "media_enc_virus"
)
function Invoke-Wsl([string]$script) { & wsl.exe -d Ubuntu-24.04 -- bash -lc $script; return $LASTEXITCODE }
foreach ($workload in $workloads) {
  Write-Host "[$(Get-Date)] $Label START $workload"
  $runCode = Invoke-Wsl "$ssh 'cd /home/ubuntu/pantheongpu && python3 pantheon.py --platform cuda --test $workload --duration 300 --gpu $GpuSelector --mem 50 --verify'"
  $runId = (& wsl.exe -d Ubuntu-24.04 -- bash -lc "$ssh 'ls -td /home/ubuntu/pantheongpu/results/* | head -1'").Trim()
  $runId = Split-Path $runId -Leaf
  $copy = "${remote}:/home/ubuntu/pantheongpu/database/pantheon_report_${runId}_0001_${workload}_gpu*.json /mnt/c/Users/sssnk/OneDrive/Documents/$Website/database/"
  $copyCode = Invoke-Wsl "$scp $copy"
  if ($copyCode -eq 0) {
    Push-Location "C:\Users\sssnk\OneDrive\Documents\$Website"
    try {
      # Pantheon <= 1.0.16 reports embed the benchmark host's hostname and IP
      # address; strip them before anything reaches the public repository.
      python website_utils/sanitize_reports.py | Select-Object -Last 1
      python website_utils/generate_web_data.py | Select-Object -Last 1
      python -m pytest -q
      if ($LASTEXITCODE -eq 0) {
        & C:\Users\sssnk\OneDrive\Documents\pantheon\.venv\Scripts\mkdocs.exe build --strict
        git add "database/pantheon_report_${runId}_0001_${workload}_gpu*.json" docs/assets/web_data.json
        git commit -m "Add $Label $workload benchmark"
        for ($attempt = 1; $attempt -le 3; $attempt++) {
          git fetch origin
          git rebase origin/main
          git push origin HEAD:main
          if ($LASTEXITCODE -eq 0) { break }
          Start-Sleep -Seconds 5
        }
      } else {
        # A failing check means the report or site is not publishable;
        # never commit past it.
        Write-Warning "$Label $workload failed website checks; skipping commit"
      }
    } finally { Pop-Location }
  } else { Write-Warning "$Label has no report files for $workload (run=$runCode copy=$copyCode)" }
  Write-Host "[$(Get-Date)] $Label DONE $workload"
}
$terminateJson = "{`"instance_ids`:[`"$InstanceId`"]}"
$terminate64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($terminateJson))
Write-Host "[$(Get-Date)] $Label queue complete; terminating $InstanceId"
$terminateCommand = 'echo ' + $terminate64 + ' | base64 -d | curl -sS --fail-with-body -H "Authorization: Bearer "' + '$(tr -d "\r\n" < /home/saqib/.ssh/lambda_keys)' + ' -H "Content-Type: application/json" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate --data-binary @- | jq -r ".data.terminated_instances[]?.id // .error.message"'
Invoke-Wsl $terminateCommand
