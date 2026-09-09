# Stop existing processes
Get-Process -Name ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$ports = @(8010, 5173)
foreach ($p in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue
    if ($conn) {
        $pids = $conn.OwningProcess | Select-Object -Unique
        foreach ($procId in $pids) {
            if ($procId -gt 0) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Start-Sleep -Seconds 1

# 1. Start Backend (Uvicorn FastAPI)
$backendPath = 'd:\myProject\Mitsuka\apps\local-api'
$pythonExe = Join-Path $backendPath '.venv\Scripts\python.exe'
Start-Process -FilePath $pythonExe -ArgumentList '-m uvicorn main:app --host 0.0.0.0 --port 8010' -WorkingDirectory $backendPath -WindowStyle Hidden

# 2. Serve the production build so the public PWA receives fresh hashed assets.
$frontendPath = 'd:\myProject\Mitsuka\airi\apps\stage-web'
Start-Process cmd.exe -ArgumentList '/c pnpm exec vite preview --host :: --port 5173' -WorkingDirectory $frontendPath -WindowStyle Hidden

# 3. Start Ngrok
Start-Process -FilePath 'ngrok.exe' -ArgumentList 'http 5173' -WorkingDirectory 'd:\myProject\Mitsuka' -WindowStyle Hidden

# 4. Wait for Ngrok URL
$url = ''
for ($i = 0; $i -lt 25; $i++) {
    Start-Sleep -Seconds 1
    try {
        $tunnels = (Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 2).tunnels
        if ($tunnels -and $tunnels.Count -gt 0) {
            $url = $tunnels[0].public_url
            break
        }
    } catch {}
}

# 5. Wait for Backend
$backendReady = 'Starting...'
for ($i = 0; $i -lt 20; $i++) {
    try {
        $res = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/health' -TimeoutSec 2
        if ($res.status -eq 'ok') {
            $backendReady = 'Online [200 OK]'
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host '========================================='
Write-Host ' Mitsuka AI Services Started Successfully'
Write-Host '========================================='
Write-Host ('Backend (8010):  ' + $backendReady)
Write-Host 'Frontend (5173): Online [Vite]'
Write-Host ('Ngrok URL:       ' + $url)
Write-Host '========================================='
