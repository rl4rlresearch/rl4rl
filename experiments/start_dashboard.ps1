param([int]$Port = 8765)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dashboardUrl = "http://127.0.0.1:$Port"
try {
    $revision = Invoke-RestMethod "$dashboardUrl/api/revision" -TimeoutSec 2
    if ($revision.revision) {
        Write-Output "Dashboard already running: $dashboardUrl"
        return
    }
} catch {}

$pythonCandidates = @(
    (Join-Path $repoRoot '.venv/Scripts/python.exe'),
    (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
)
$pythonPath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonPath) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
$logRoot = Join-Path $repoRoot 'outputs/dashboard'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$dashboardProcess = Start-Process -FilePath $pythonPath -ArgumentList @(
    '-u', 'experiments/live_trajectory_dashboard.py', '--port', $Port
) -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'stderr.log')
$dashboardProcess.Id | Set-Content -LiteralPath (Join-Path $logRoot 'server.pid')
Write-Output "Dashboard started (PID $($dashboardProcess.Id)): $dashboardUrl"
