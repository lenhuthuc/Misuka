import os, sys, subprocess, time, json, urllib.request


def stop_listeners(ports: set[int]) -> None:
    """Stop only valid PIDs listening on the requested TCP ports.

    The former `cmd.exe for /f` expression used the interactive `%a` syntax.
    Under `subprocess(..., shell=True)`, CMD expands it before the loop runs,
    producing `taskkill /PID 0` repeatedly instead of a real listener PID.
    """
    netstat = subprocess.run(
        ["netstat", "-aon", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    for line in netstat.stdout.splitlines():
        columns = line.split()
        if len(columns) < 5 or columns[0] != "TCP":
            continue
        try:
            port = int(columns[1].rsplit(":", 1)[1])
            pid = int(columns[-1])
        except (IndexError, ValueError):
            continue
        if port in ports and pid > 0:
            pids.add(pid)

    for pid in pids:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


# Stop existing processes without invoking CMD or killing PID 0.
for image_name in ("cloudflared.exe", "ngrok.exe"):
    subprocess.run(
        ["taskkill", "/F", "/IM", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
stop_listeners({8010, 5173})

time.sleep(1)

backend_dir = r'd:\myProject\Mitsuka\apps\local-api'
python_exe = os.path.join(backend_dir, '.venv', 'Scripts', 'python.exe')
frontend_dir = r'd:\myProject\Mitsuka\airi'
cloudflared_exe = r'C:\Program Files (x86)\cloudflared\cloudflared.exe'
cloudflared_cfg = r'C:\Users\admin\.cloudflared\config.yml'

os.makedirs(r'd:\myProject\Mitsuka\logs', exist_ok=True)
backend_log = open(r'd:\myProject\Mitsuka\logs\backend.log', 'w', encoding='utf-8')
frontend_log = open(r'd:\myProject\Mitsuka\logs\frontend.log', 'w', encoding='utf-8')
tunnel_log = open(r'd:\myProject\Mitsuka\logs\tunnel.log', 'w', encoding='utf-8')

# 1. Start Backend (Uvicorn FastAPI)
p_backend = subprocess.Popen([python_exe, '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8010'], cwd=backend_dir, stdout=backend_log, stderr=backend_log, stdin=subprocess.DEVNULL)

# 2. Serve the production build. This also lets the live PWA receive the new
# service worker and asset hashes instead of staying on a cached dev snapshot.
p_frontend = subprocess.Popen('pnpm exec vite preview --host :: --port 5173', cwd=os.path.join(frontend_dir, 'apps', 'stage-web'), shell=True, stdout=frontend_log, stderr=frontend_log, stdin=subprocess.DEVNULL)

# 3. Start Cloudflare Tunnel for haquason.uk
p_tunnel = subprocess.Popen([cloudflared_exe, '--config', cloudflared_cfg, 'tunnel', 'run'], stdout=tunnel_log, stderr=tunnel_log, stdin=subprocess.DEVNULL)

print('Services launched. Initializing Backend, Frontend, and Cloudflare Tunnel...')
sys.stdout.flush()

# 4. Wait for Backend
for _ in range(40):
    time.sleep(1)
    try:
        req = urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2)
        if json.loads(req.read().decode('utf-8')).get('status') == 'ok':
            print('Backend (8010):  Online [200 OK]')
            sys.stdout.flush()
            break
    except Exception:
        pass

# 5. Wait for Frontend
for _ in range(35):
    try:
        req = urllib.request.urlopen('http://127.0.0.1:5173/', timeout=2)
        if req.status == 200:
            print('Frontend (5173): Online [Vite 200 OK]')
            sys.stdout.flush()
            break
    except Exception:
        time.sleep(1)

# 6. Wait for Domain haquason.uk
for _ in range(30):
    time.sleep(1)
    try:
        req = urllib.request.Request('https://haquason.uk/health', headers={'User-Agent': 'HealthCheck/1.0'})
        res = urllib.request.urlopen(req, timeout=3)
        if json.loads(res.read().decode('utf-8')).get('status') == 'ok':
            print('Domain:          https://haquason.uk/ Online [200 OK]')
            break
    except Exception:
        pass

print('=' * 45)
print('  Mitsuka AI Services are LIVE at https://haquason.uk/')
print('=' * 45)
sys.stdout.flush()

# Keep supervisor alive
while True:
    time.sleep(3600)
