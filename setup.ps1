# =============================================================
# setup.ps1 - Chay MOT LAN DUY NHAT
# Tu dong: tao venv 3.12, cai packages, cai Silver Sponsor
# =============================================================
param(
    [switch]$Force  # them -Force de tao lai venv tu dau
)

$ProjectRoot = $PSScriptRoot
$VenvPath    = "$ProjectRoot\venv"
$PythonExe   = ""
$LogFile     = "$ProjectRoot\setup.log"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "  [$n/$total] $msg" -ForegroundColor Cyan
}

function OK($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function WARN($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }
function ERR($msg)  { Write-Host "  XX  $msg" -ForegroundColor Red }

Clear-Host
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "   Super AI Quant VN - One-Time Setup" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Log "=== Setup bat dau ==="

# ── 0. Tim Python 3.12 ────────────────────────────────────────
Step 0 6 "Tim Python 3.12..."

$candidates = @(
    (Get-Command "py" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe"
) | Where-Object { $_ -and (Test-Path $_) }

foreach ($candidate in $candidates) {
    try {
        if ($candidate -match "\\py\.exe$") {
            $ver = & $candidate -3.12 --version 2>&1
            if ($ver -match "3\.12") {
                # Lay duong dan python.exe thuc su
                $PythonExe = & $candidate -3.12 -c "import sys; print(sys.executable)" 2>&1
                break
            }
        } else {
            $ver = & $candidate --version 2>&1
            if ($ver -match "3\.12") {
                $PythonExe = $candidate
                break
            }
        }
    } catch {}
}

if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
    # Thu py launcher
    try {
        $ver = & py -3.12 --version 2>&1
        if ($ver -match "3\.12") {
            $PythonExe = & py -3.12 -c "import sys; print(sys.executable)" 2>&1
        }
    } catch {}
}

if (-not $PythonExe) {
    ERR "Khong tim thay Python 3.12!"
    ERR "Tai tai: https://www.python.org/downloads/release/python-3120/"
    exit 1
}
OK "Python 3.12: $PythonExe"
Log "Python: $PythonExe"

# ── 1. Tao / kiem tra venv ────────────────────────────────────
Step 1 6 "Chuan bi Virtual Environment..."

if ($Force -and (Test-Path $VenvPath)) {
    WARN "Xoa venv cu (-Force)..."
    Remove-Item -Recurse -Force $VenvPath
}

if (Test-Path "$VenvPath\Scripts\python.exe") {
    $existingVer = & "$VenvPath\Scripts\python.exe" --version 2>&1
    if ($existingVer -match "3\.12") {
        OK "Venv 3.12 da ton tai: $VenvPath"
    } else {
        WARN "Venv hien tai khong phai 3.12 ($existingVer) - tao lai..."
        Remove-Item -Recurse -Force $VenvPath
        & $PythonExe -m venv $VenvPath
    }
} else {
    Log "Tao venv moi tai $VenvPath"
    & $PythonExe -m venv $VenvPath
}

if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    ERR "Tao venv that bai!"
    exit 1
}

$VenvPython = "$VenvPath\Scripts\python.exe"
$VenvPip    = "$VenvPath\Scripts\pip.exe"
$VenvVer    = & $VenvPython --version 2>&1
OK "Venv san sang: $VenvVer"

# ── 2. Nang cap pip ───────────────────────────────────────────
Step 2 6 "Nang cap pip..."
& $VenvPip install --upgrade pip -q
OK "pip da cap nhat"

# ── 3. Cai requirements.txt ───────────────────────────────────
Step 3 6 "Cai requirements.txt..."

if (-not (Test-Path "$ProjectRoot\requirements.txt")) {
    ERR "Khong tim thay requirements.txt!"
    exit 1
}

$output = & $VenvPip install -r "$ProjectRoot\requirements.txt" 2>&1
$output | ForEach-Object { Log $_ }

# Kiem tra cac package core
$corePackages = @("streamlit","pandas","numpy","tigramite","openai","anthropic","dashscope")
$missingCore  = @()
foreach ($pkg in $corePackages) {
    $check = & $VenvPython -c "import $($pkg.Replace('-','_'))" 2>&1
    if ($LASTEXITCODE -ne 0) { $missingCore += $pkg }
}

if ($missingCore.Count -gt 0) {
    WARN "Cai lai cac package bi loi: $($missingCore -join ', ')"
    & $VenvPip install @missingCore
}
OK "Requirements da cai"

# ── 4. Cai extra dependencies ────────────────────────────────
Step 4 6 "Cai extra dependencies (html2text, flask, lxml, schedule)..."
& $VenvPip install html2text flask lxml schedule -q
OK "Extra dependencies da cai"

# ── 5. Cai Silver Sponsor (vnstock_data, ta, news) ────────────
Step 5 6 "Cai Silver Sponsor packages (vnstock_data, vnstock_ta, vnstock_news)..."

# Kiem tra da co chua
$sponsorOK = $true
foreach ($pkg in @("vnstock_data","vnstock_ta","vnstock_news")) {
    $check = & $VenvPython -c "import $pkg" 2>&1
    if ($LASTEXITCODE -ne 0) { $sponsorOK = $false; break }
}

if ($sponsorOK) {
    OK "Silver Sponsor packages da duoc cai truoc do"
} else {
    WARN "Chua co Silver Sponsor packages - chay vnstock-installer..."
    Write-Host ""
    Write-Host "  >>> HUONG DAN <<<" -ForegroundColor Yellow
    Write-Host "  Trinh duyet se mo ra - dang nhap tai khoan Sponsor" -ForegroundColor White
    Write-Host "  Dien venv path: $VenvPath" -ForegroundColor White
    Write-Host "  Chon Python: $VenvPython" -ForegroundColor White
    Write-Host ""

    # Cai vnstock-installer vao venv truoc
    & $VenvPip install vnstock-installer -q

    # Chay installer tu dung venv
    & $VenvPython -m vnstock_installer

    # Kiem tra lai sau khi install
    $sponsorOK = $true
    foreach ($pkg in @("vnstock_data","vnstock_ta","vnstock_news")) {
        $check = & $VenvPython -c "import $pkg" 2>&1
        if ($LASTEXITCODE -ne 0) {
            $sponsorOK = $false
            WARN "Van thieu: $pkg"
        }
    }
    if ($sponsorOK) {
        OK "Silver Sponsor packages da cai thanh cong"
    } else {
        WARN "Mot so Sponsor package chua cai - chay lai setup.ps1 sau khi cai"
    }
}

# ── 6. Kiem tra cuoi ─────────────────────────────────────────
Step 6 6 "Kiem tra toan bo imports..."

$checks = @{
    "vnstock"       = "import vnstock"
    "pandas"        = "import pandas"
    "numpy"         = "import numpy"
    "streamlit"     = "import streamlit"
    "tigramite"     = "import tigramite"
    "openai"        = "import openai"
    "anthropic"     = "import anthropic"
    "dashscope"     = "import dashscope"
    "vnstock_data"  = "import vnstock_data"
    "vnstock_ta"    = "import vnstock_ta"
    "vnstock_news"  = "import vnstock_news"
}

$allOK = $true
foreach ($name in $checks.Keys) {
    $res = & $VenvPython -c $checks[$name] 2>&1
    if ($LASTEXITCODE -eq 0) {
        OK "$name"
    } else {
        WARN "$name - CHUA CO"
        $allOK = $false
    }
}

# ── Kiem tra .env ─────────────────────────────────────────────
Write-Host ""
if (-not (Test-Path "$ProjectRoot\.env")) {
    if (Test-Path "$ProjectRoot\.env.example") {
        Copy-Item "$ProjectRoot\.env.example" "$ProjectRoot\.env"
        WARN ".env da tao tu .env.example - dien API keys vao file .env!"
    } else {
        WARN "Chua co file .env - tao va dien API keys!"
    }
} else {
    OK ".env da ton tai"
}

# ── Tao run.ps1 neu chua co ───────────────────────────────────
if (-not (Test-Path "$ProjectRoot\run.ps1")) {
    WARN "Khong tim thay run.ps1 - vui long download tu artifacts"
}

# ── Ket qua ───────────────────────────────────────────────────
Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
if ($allOK) {
    Write-Host "   SETUP HOAN TAT!" -ForegroundColor Green
    Write-Host "   Chay app bang: .\run.ps1" -ForegroundColor Green
} else {
    Write-Host "   SETUP XONG (mot so package chua co)" -ForegroundColor Yellow
    Write-Host "   Kiem tra log: $LogFile" -ForegroundColor Yellow
    Write-Host "   Chay lai: .\setup.ps1" -ForegroundColor Yellow
}
Write-Host "======================================================" -ForegroundColor Cyan
Log "=== Setup ket thuc ==="
