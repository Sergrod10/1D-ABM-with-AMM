$baseUrl = "https://data.binance.vision/data/spot/daily/klines/ETHUSDT/1s"
$outDir = ".\dataset"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$start = Get-Date "2025-03-24"
$end   = Get-Date "2026-03-24"

for ($d = $start; $d -le $end; $d = $d.AddDays(1)) {
    $dateStr = $d.ToString("yyyy-MM-dd")
    $fileName = "$symbol-$interval-$dateStr.zip"
    $url = "$baseUrl/$fileName"
    $outFile = Join-Path $outDir $fileName
    if (Test-Path $outFile) {
        Write-Host "[SKIP] $fileName"
        continue
    }
    Write-Host "[GET ] $fileName"
    try {
        Invoke-WebRequest -Uri $url -OutFile $outFile
        Write-Host "[OK  ] $fileName"
    }
    catch {
        Write-Host "[FAIL] $fileName"
    }
}