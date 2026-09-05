param([string]$Campaign = 'data/c0c3/tiny-kws-rnn-openevolve-v2-1-cpu-campaign')
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location -LiteralPath $Repo
$Python = (Resolve-Path 'outputs/tiny-seed-search/venv/Scripts/python.exe').Path
$Campaign = (Resolve-Path -LiteralPath $Campaign).Path
$PythonCampaign = '\\?\' + $Campaign
$Codex = (Get-Command codex -ErrorAction Stop).Source
$env:PYTHONPATH = $Repo
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$Task = Get-Content "$Campaign/inputs/task.json" -Raw | ConvertFrom-Json
if ($Task.preferred_backend -ne 'hybrid_modal' -or $Task.extension_options.modal_app -ne 'rl4rl-tiny-kws-cpu') {
    throw 'Apply and validate the recorded TinyKWS Modal amendment before resuming.'
}
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$Receipt = "$Campaign/windows-resume-$Stamp.json"
$Rows = Get-Content "$Campaign/schedule.json" -Raw | ConvertFrom-Json
$LogRoot = "$Campaign/workers"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$Workers = @()
foreach ($Row in $Rows) {
    if ($Row.condition -notin @('C0','C1','C2','C3')) { continue }
    $State = Get-Content "$Campaign/runs/$($Row.run_id)/state.json" -Raw | ConvertFrom-Json
    if ($State.proposals_used -ge 200) { continue }
    if ($State.active) { throw "Unrecovered active proposal in $($Row.run_id)" }
    $Log = "$LogRoot/b$('{0:d2}' -f [int]$Row.block)-$($Row.condition)-$Stamp"
    $Arguments = @('-u', '-m', 'experiments.c0c3_factorial.cli', 'resume-staged-trajectory',
        '--campaign', ('"'+$PythonCampaign+'"'), '--run-id', $Row.run_id,
        '--python-bin', ('"'+$Python+'"'), '--codex-binary', ('"'+$Codex+'"'))
    $Worker = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Log.stdout.log" -RedirectStandardError "$Log.stderr.log"
    $Workers += [pscustomobject]@{run_id=$Row.run_id;pid=$Worker.Id;starting_proposals=$State.proposals_used;log_prefix=$Log;started_at=(Get-Date).ToUniversalTime().ToString('o')}
    ConvertTo-Json -InputObject $Workers -Depth 5 | Set-Content -LiteralPath $Receipt -Encoding utf8
}
$Workers | Format-Table
Write-Output "Receipt: $Receipt"
