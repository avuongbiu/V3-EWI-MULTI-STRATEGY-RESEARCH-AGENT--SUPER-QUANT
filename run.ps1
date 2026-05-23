# =============================================================
# run.ps1 - Chay hang ngay
# Tu dong: kiem tra venv, package, .env -> khoi dong Streamlit
# =============================================================

$ProjectRoot = $PSScriptRoot
$VenvPath    = "$ProjectRoot\venv"
$VenvPython  = "$VenvPath\Scripts\python.exe"
$VenvPip     = "$VenvPath\Scripts\pip.exe"
$Streamlit   = "$VenvPath\Scripts\streamlit.exe"

function OK($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function WARN($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }
function ERR($msg)  { Write-Host "  XX  $msg" -ForegroundColor Red }

Clear-Host
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "   Super AI Quant VN" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# ── 1. Kiem tra venv ─────────────────────────────────────────
if (-not (Test-Path $VenvPython)) {
    ERR "Chua co venv! Chay .\setup.ps1 truoc."
    exit 1
}

$ver = & $VenvPython --version 2>&1
if ($ver -notmatch "3\.12") {
    WARN "Venv dang chay $ver (khuyen nghi 3.12)"
    WARN "Chay .\setup.ps1 -Force de tao lai"
} else {
    OK "Python $ver"
}

# ── 2. Kiem tra .env ─────────────────────────────────────────
if (-not (Test-Path "$ProjectRoot\.env")) {
    ERR "Chua co file .env!"
    ERR "Copy .env.example thanh .env va dien API keys"
    $ans = Read-Host "Tao .env trong? (y/n)"
    if ($ans -eq 'y') {
        if (Test-Path "$ProjectRoot\.env.example") {
            Copy-Item "$ProjectRoot\.env.example" "$ProjectRoot\.env"
            WARN ".env da tao - dien API keys vao: $ProjectRoot\.env"
            notepad "$ProjectRoot\.env"
            Read-Host "Nhan Enter sau khi da dien API keys"
        }
    }
} else {
    OK ".env da co"
}

# ── 3. Kiem tra nhanh core packages ──────────────────────────
Write-Host ""
Write-Host "  Kiem tra packages..." -ForegroundColor Cyan

$critical = @("streamlit","pandas","numpy","vnstock")
$missing  = @()
foreach ($pkg in $critical) {
    $res = & $VenvPython -c "import $($pkg.Replace('-','_'))" 2>&1
    if ($LASTEXITCODE -ne 0) { $missing += $pkg }
}

if ($missing.Count -gt 0) {
    WARN "Thieu packages: $($missing -join ', ') - dang cai..."
    & $VenvPip install @missing -q
    OK "Da cai xong"
} else {
    OK "Tat ca packages san sang"
}

# ── 4. Kiem tra Silver Sponsor ───────────────────────────────
$sponsorStatus = @()
foreach ($pkg in @("vnstock_data","vnstock_ta","vnstock_news")) {
    $res = & $VenvPython -c "import $pkg" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $sponsorStatus += "$pkg OK"
    } else {
        $sponsorStatus += "$pkg MISSING"
    }
}
Write-Host "  Silver Sponsor: $($sponsorStatus -join ' | ')" -ForegroundColor $(
    if ($sponsorStatus -join '' -match 'MISSING') { 'Yellow' } else { 'Green' }
)

# ── 5. Chay Streamlit ────────────────────────────────────────
Write-Host ""
Write-Host "  Khoi dong..." -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $Streamlit)) {
    & $VenvPip install streamlit -q
}

& $Streamlit run "$ProjectRoot\app.py" `
    --server.headless false `
    --browser.gatherUsageStats false `
    --theme.base dark
