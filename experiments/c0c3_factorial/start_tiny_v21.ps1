param(
    [string]$Campaign = 'data/c0c3/tiny-v21',
    [string]$Python = 'outputs/tiny-seed-search/venv/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $RepoRoot
$Campaign = (Resolve-Path -LiteralPath $Campaign).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
$Codex = (Get-Command codex -ErrorAction Stop).Source
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = $RepoRoot
$Protocol = Get-Content -LiteralPath (Join-Path $Campaign 'inputs/protocol.json') -Raw | ConvertFrom-Json
$Task = Get-Content -LiteralPath (Join-Path $Campaign 'inputs/task.json') -Raw | ConvertFrom-Json
$Schedule = @(Get-Content -LiteralPath (Join-Path $Campaign 'schedule.json') -Raw | ConvertFrom-Json)
if ($Protocol.include_c4 -ne $false -or $Protocol.blocks -ne 5 -or $Protocol.budget.proposals -ne 200 -or $Schedule.Count -ne 20) {
    throw 'Expected five blocks, C0-C3 only, 200 proposals, and twenty trajectories.'
}
if ($Task.adapter -ne 'tiny_adderboard_v21' -or $Task.extension_options.unlimited_subject_workers -ne $true -or $Task.extension_options.local_evaluator_capacity -ne 3) {
    throw 'Unexpected task or concurrency settings.'
}
$LaunchFile = Join-Path $Campaign 'windows-launch.json'
if (Test-Path -LiteralPath $LaunchFile) { throw 'A launch receipt already exists. Inspect current workers before relaunching.' }
& $Python -m experiments.c0c3_factorial.cli validate --campaign $Campaign
if ($LASTEXITCODE -ne 0) { throw 'Campaign launch validation failed.' }
$LogRoot = Join-Path $Campaign 'workers'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Workers = @()
foreach ($Row in $Schedule) {
    if ($Row.condition -notin @('C0', 'C1', 'C2', 'C3')) { throw 'Unexpected condition.' }
    $Arguments = @('-u', '-m', 'experiments.c0c3_factorial.cli', 'start-staged-trajectory',
        '--campaign', ('"' + $Campaign + '"'), '--run-id', $Row.run_id,
        '--python-bin', ('"' + $Python + '"'), '--codex-binary', ('"' + $Codex + '"'))
    $Worker = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogRoot ($Row.run_id + '.stdout.log')) `
        -RedirectStandardError (Join-Path $LogRoot ($Row.run_id + '.stderr.log'))
    $Workers += [pscustomobject]@{run_id=$Row.run_id; pid=$Worker.Id; block=$Row.block; condition=$Row.condition}
    [pscustomobject]@{started_at=(Get-Date).ToUniversalTime().ToString('o'); workers=$Workers; modal_cpu_per_evaluator=2; modal_memory_mib_per_evaluator=4096; modal_evaluator_cap=$null; codex_worker_cap=$null; windows_evaluator_cap=3} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $LaunchFile -Encoding UTF8
}
$Workers | Format-Table
