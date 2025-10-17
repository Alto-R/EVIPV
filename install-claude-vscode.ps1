[CmdletBinding()]
param(
  [Parameter(Mandatory=$false)][string]$Token,
  [Parameter(Mandatory=$false)][string]$BaseUrl,
  [switch]$Yes,
  [switch]$Reinstall
)

$script:AutoApprove = $Yes.IsPresent
$script:Reinstall = $Reinstall.IsPresent
$script:ExtId = 'anthropic.claude-code'
$script:CodeCLIResolved = ''

function Test-Command {
  param([string]$Name)
  try { Get-Command $Name -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

function Confirm-Action {
  param([string]$Prompt)
  if ($script:AutoApprove) { return $true }
  $resp = Read-Host "$Prompt [y/N]"
  return ($resp -match '^(?i)(y|yes)$')
}

function Normalize-BaseUrl {
  param([string]$Url)
  if ([string]::IsNullOrWhiteSpace($Url)) { return $Url }
  $u = $Url.TrimEnd('/')
  return $u
}

function Get-CodeCLIPath {
  if (Test-Command 'code') { return (Get-Command code).Source }
  if (Test-Command 'code.cmd') { return (Get-Command code.cmd).Source }
  $candidates = @()
  if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Microsoft VS Code\bin\code.cmd') }
  if ($env:ProgramFiles) { $candidates += (Join-Path $env:ProgramFiles 'Microsoft VS Code\bin\code.cmd') }
  $pf86 = ${env:ProgramFiles(x86)}
  if ($pf86) { $candidates += (Join-Path $pf86 'Microsoft VS Code\bin\code.cmd') }
  foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
  return $null
}

function Write-CodeDetection {
  $code = Get-CodeCLIPath
  if ($code) {
    $ver = ''
    try { $ver = (& $code --version | Select-Object -First 1) } catch { }
    if ($ver) {
      Write-Host "VS Code detected: $ver ($code)" -ForegroundColor Green
    } else {
      Write-Host "VS Code CLI detected at: $code" -ForegroundColor Green
    }
  } else {
    Write-Host 'VS Code not detected; will attempt installation if needed.' -ForegroundColor Yellow
  }
}

function Invoke-WingetInstall {
  param([string]$PackageId)
  $args = @('install','-e','--id',$PackageId,'--accept-package-agreements','--accept-source-agreements','--scope','user')
  & winget @args
  return $LASTEXITCODE
}

function Ensure-VSCode {
  $code = Get-CodeCLIPath
  if ($code) { return $code }
  if (Test-Command 'winget') {
    if (-not (Confirm-Action 'Install VS Code via winget?')) { return $null }
    Write-Host "Installing Visual Studio Code via winget..." -ForegroundColor Yellow
    $exit = Invoke-WingetInstall -PackageId 'Microsoft.VisualStudioCode'
    if ($exit -ne 0) {
      Write-Host "winget exited with $exit; retrying with output..." -ForegroundColor Yellow
      & winget install -e --id Microsoft.VisualStudioCode --accept-package-agreements --accept-source-agreements --scope user
    }
    Start-Sleep -Seconds 2
    return (Get-CodeCLIPath)
  }

  # Fallback: direct download installer
  if (-not (Confirm-Action 'winget not found. Download VS Code User Installer and install now?')) { return $null }
  $arch = 'win32-x64-user'
  try {
    $pa = ($env:PROCESSOR_ARCHITECTURE + '').ToLower()
    if ($pa -eq 'arm64') { $arch = 'win32-arm64-user' }
  } catch { }
  $url = "https://update.code.visualstudio.com/latest/$arch/stable"
  $temp = Join-Path $env:TEMP 'VSCodeUserSetup.exe'
  Write-Host "Downloading $url ..." -ForegroundColor Yellow
  try {
    Invoke-WebRequest -Uri $url -OutFile $temp -UseBasicParsing
  } catch {
    Write-Error "Failed to download VS Code from $url. $_"; return $null
  }
  Write-Host "Running installer silently..." -ForegroundColor Yellow
  try {
    Start-Process -FilePath $temp -ArgumentList '/VERYSILENT','/MERGETASKS=addtopath' -Wait -NoNewWindow
  } catch {
    Write-Error "Failed to run VS Code installer. $_"; return $null
  }
  Start-Sleep -Seconds 2
  return (Get-CodeCLIPath)
}

function Reinstall-VSCode {
  if (Test-Command 'winget') {
    if (-not (Confirm-Action 'Reinstall/upgrade VS Code via winget?')) { return }
    Write-Host "Upgrading/Reinstalling Visual Studio Code via winget..." -ForegroundColor Yellow
    & winget upgrade -e --id Microsoft.VisualStudioCode --accept-package-agreements --accept-source-agreements --scope user
    if ($LASTEXITCODE -ne 0) {
      Write-Host "winget upgrade failed or not applicable; trying install to force repair/upgrade..." -ForegroundColor Yellow
      & winget install -e --id Microsoft.VisualStudioCode --accept-package-agreements --accept-source-agreements --scope user
    }
    Start-Sleep -Seconds 2
    return
  }
  if (-not (Confirm-Action 'winget not found. Download latest VS Code User Installer to upgrade now?')) { return }
  $arch = 'win32-x64-user'
  try { if ((($env:PROCESSOR_ARCHITECTURE + '').ToLower()) -eq 'arm64') { $arch = 'win32-arm64-user' } } catch { }
  $url = "https://update.code.visualstudio.com/latest/$arch/stable"
  $temp = Join-Path $env:TEMP 'VSCodeUserSetup.exe'
  Write-Host "Downloading $url ..." -ForegroundColor Yellow
  try { Invoke-WebRequest -Uri $url -OutFile $temp -UseBasicParsing } catch { Write-Error "Failed to download VS Code from $url. $_"; return }
  Write-Host "Running installer silently..." -ForegroundColor Yellow
  try { Start-Process -FilePath $temp -ArgumentList '/VERYSILENT','/MERGETASKS=addtopath' -Wait -NoNewWindow } catch { Write-Error "Failed to run VS Code installer. $_"; return }
  Start-Sleep -Seconds 2
}

function Install-Extension {
  $code = Get-CodeCLIPath
  if (-not $code) { $code = Ensure-VSCode }
  if (-not $code) { Write-Error "VS Code CLI not found; cannot install extension."; exit 20 }
  Write-Host "Using VS Code CLI: $code" -ForegroundColor Green
  $script:CodeCLIResolved = $code
  if ($script:Reinstall) {
    Write-Host "Reinstalling VS Code extension: $($script:ExtId)" -ForegroundColor Yellow
    & $code --uninstall-extension $script:ExtId 2>$null | Out-Null
  } else {
    # Detect if already installed (case-insensitive)
    $list = & $code --list-extensions 2>$null
    if ($list) {
      $exists = $false
      foreach ($line in $list) { if ($line.Trim().ToLower() -eq $script:ExtId.ToLower()) { $exists = $true; break } }
      if ($exists) { Write-Host "Extension '$($script:ExtId)' already installed. Skipping." -ForegroundColor Green; return }
    }
    Write-Host "Installing VS Code extension: $($script:ExtId)" -ForegroundColor Yellow
  }
  & $code --install-extension $script:ExtId --force
  if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install extension '$($script:ExtId)'."; exit 21 }
}

function Prompt-IfNeeded {
  if (-not $Token) {
    $sec = Read-Host -AsSecureString -Prompt 'Enter ANTHROPIC_AUTH_TOKEN'
    $cred = New-Object System.Net.NetworkCredential('', $sec)
    $Token = $cred.Password
    if (-not $Token) { Write-Error 'Token is required.'; exit 2 }
  }
  if (-not $BaseUrl) {
    $BaseUrl = Read-Host 'Enter ANTHROPIC_BASE_URL (e.g. https://api.anthropic.com)'
    if (-not $BaseUrl) { Write-Error 'ANTHROPIC_BASE_URL is required.'; exit 2 }
  }
}

function Write-ClaudeFiles {
  $Base = Normalize-BaseUrl $BaseUrl
  $targetDir = Join-Path $HOME '.claude'
  $settingsFile = Join-Path $targetDir 'settings.json'
  $configFile = Join-Path $targetDir 'config.json'
  if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir -Force | Out-Null }

  $settings = @"
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "$Token",
    "ANTHROPIC_BASE_URL": "$Base",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "API_TIMEOUT_MS": "600000",
    "BASH_DEFAULT_TIMEOUT_MS": "600000",
    "BASH_MAX_TIMEOUT_MS": "600000",
    "MCP_TIMEOUT": "30000",
    "MCP_TOOL_TIMEOUT": "600000",
    "CLAUDE_API_TIMEOUT": "600000"
  },
  "permissions": {
    "allow": [],
    "deny": []
  }
}
"@.Trim()

  $config = '{"primaryApiKey": "default"}'

  Set-Content -Path $settingsFile -Value $settings -Encoding ascii
  Set-Content -Path $configFile -Value $config -Encoding ascii
  try {
    (Get-Item $settingsFile).Attributes = 'Normal'
    (Get-Item $configFile).Attributes = 'Normal'
  } catch { }
  Write-Host "✅ Wrote $settingsFile"
  Write-Host "✅ Wrote $configFile"
}

# Env fallback
if (-not $Token) {
  $Token = $env:ANTHROPIC_AUTH_TOKEN
}
if (-not $BaseUrl) { $BaseUrl = $env:ANTHROPIC_BASE_URL }

Write-CodeDetection
if ($script:Reinstall) {
  # Attempt to reinstall/upgrade VS Code first
  $null = Reinstall-VSCode
}
Install-Extension
if (-not $script:AutoApprove) {
  Prompt-IfNeeded
  Write-Host "About to write ~/.claude settings and config." -ForegroundColor Yellow
  if (-not (Confirm-Action 'Proceed?')) { Write-Host 'Aborted.'; exit 3 }
} else {
  if (-not $Token -or -not $BaseUrl) { Write-Error 'In non-interactive mode, --Token and --BaseUrl are required.'; exit 2 }
}
Write-ClaudeFiles
Write-Host "✅ Extension '$($script:ExtId)' installed." -ForegroundColor Green
Write-Host "   ANTHROPIC_AUTH_TOKEN=$Token"
Write-Host "   ANTHROPIC_BASE_URL=$BaseUrl"
if ($script:CodeCLIResolved) { Write-Host "   VS Code CLI=$($script:CodeCLIResolved)" }
Write-Host "✅ Configuration complete." -ForegroundColor Green
