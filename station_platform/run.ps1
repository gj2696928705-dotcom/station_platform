param(
    [ValidateSet('setup','start','stop','status','center','node','test')][string]$Command = 'start',
    [ValidateSet('center','node')][string]$Only,
    [int]$CenterPort = 8780,
    [int]$NodePort = 8781,
    [string]$Python = 'D:\Anaconda3\anaconda\python.exe'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskVenv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskPreviousTemp = $env:TEMP
$taskPreviousTmp = $env:TMP
$taskPreviousPipCache = $env:PIP_CACHE_DIR
$taskPreviousEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Push-Location -LiteralPath $taskRoot
try {
    if ($Command -eq 'setup') {
        $taskTemp = Join-Path $PSScriptRoot 'data\setup-temp'
        New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
        $env:TEMP = $taskTemp
        $env:TMP = $taskTemp
        $env:PIP_CACHE_DIR = Join-Path $taskTemp 'pip-cache'
        & $Python -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw '独立环境创建失败。' }
        $taskRequirements = Join-Path $PSScriptRoot 'requirements-lock.txt'
        if (-not (Test-Path -LiteralPath $taskRequirements)) { $taskRequirements = Join-Path $PSScriptRoot 'requirements.txt' }
        & $taskVenv -m pip install -r $taskRequirements
        if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
    } elseif ($Command -eq 'test') {
        $env:TEMP = Join-Path $PSScriptRoot 'data\setup-temp'
        $env:TMP = $env:TEMP
        & $taskVenv -m pytest 'station_platform/tests' -q
        if ($LASTEXITCODE -ne 0) { throw '测试未通过。' }
    } else {
        if (-not (Test-Path -LiteralPath $taskVenv)) { throw '请先运行 .\station_platform\run.ps1 -Command setup' }
        $taskArgs = @('-X','utf8','-m','station_platform',$Command,'--center-port',"$CenterPort",'--node-port',"$NodePort")
        if ($Only) { $taskArgs += @('--only',$Only) }
        & $taskVenv @taskArgs
        if ($LASTEXITCODE -ne 0) { throw '平台命令失败，请查看服务日志。' }
    }
} finally {
    $env:TEMP = $taskPreviousTemp
    $env:TMP = $taskPreviousTmp
    $env:PIP_CACHE_DIR = $taskPreviousPipCache
    [Console]::OutputEncoding = $taskPreviousEncoding
    Pop-Location
}
