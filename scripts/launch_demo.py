"""Start the read-only API and the React dev server for the demo, wait until both answer, print the URLs, and
shut both down cleanly on Ctrl-C.

Environment / flags
  API_PORT       API port (default 8000). If it is busy the script stops and tells you how to choose another:
                 `API_PORT=8765 make demo` (the frontend is started with VITE_API_URL set to match automatically;
                 if you start the frontend yourself, put VITE_API_URL=http://localhost:8765 in frontend/.env.local).
  FRONTEND_PORT  Vite port (default 5173).
  MARKET_MAKER_DB_PATH / MARKET_MAKER_ARTIFACTS_ROOT  which results the API serves (defaults: artifacts/).
  --api-only     do not start the frontend. --dry-run  print the commands and exit.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for(url: str, proc: subprocess.Popen, timeout: float, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"ERROR: {label} exited early with code {proc.returncode}. See its output above.", file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:  # noqa: BLE001 - server not up yet; retry until the deadline
            time.sleep(0.5)
    print(f"ERROR: {label} did not answer at {url} within {timeout:.0f}s.", file=sys.stderr)
    return False


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=8)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api_port = int(os.environ.get("API_PORT", "8000"))
    fe_port = int(os.environ.get("FRONTEND_PORT", "5173"))
    api_url = f"http://localhost:{api_port}"
    api_cmd = [sys.executable, "-m", "uvicorn", "market_maker.api.main:app", "--port", str(api_port)]
    fe_cmd = ["npm", "run", "dev", "--", "--port", str(fe_port), "--strictPort"]
    fe_env = dict(os.environ, VITE_API_URL=api_url)  # process env beats frontend/.env.local in Vite
    if args.dry_run:
        print("API:      " + " ".join(api_cmd))
        if not args.api_only:
            print(f"Frontend: (cd frontend && VITE_API_URL={api_url} " + " ".join(fe_cmd) + ")")
        return 0
    if port_busy(api_port):
        print(f"ERROR: port {api_port} is already in use (another API, or Docker). Pick another port, e.g.\n"
              f"    API_PORT=8765 make demo\n"
              f"The frontend is started with VITE_API_URL set to match. If you run the frontend by hand, set\n"
              f"    VITE_API_URL=http://localhost:8765\n"
              f"in frontend/.env.local. To see what uses the port: lsof -i :{api_port}", file=sys.stderr)
        return 2
    if not args.api_only:
        if not (ROOT / "frontend" / "node_modules").is_dir():
            print("ERROR: frontend dependencies are not installed. Run `make install-frontend` first.", file=sys.stderr)
            return 2
        if port_busy(fe_port):
            print(f"ERROR: frontend port {fe_port} is in use. Use FRONTEND_PORT=5174 make demo", file=sys.stderr)
            return 2
    api = fe = None
    # also honor SIGTERM and a SIGINT that a non-interactive parent shell may have set to "ignore"
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        api = subprocess.Popen(api_cmd, cwd=ROOT, start_new_session=True)
        if not wait_for(f"{api_url}/health", api, 40, "API"):
            return 1
        if not args.api_only:
            fe = subprocess.Popen(fe_cmd, cwd=ROOT / "frontend", env=fe_env, start_new_session=True)
            if not wait_for(f"http://localhost:{fe_port}", fe, 90, "frontend dev server"):
                return 1
        print("\n" + "=" * 60)
        print(f"API:       {api_url}/docs   (health: {api_url}/health)")
        if not args.api_only:
            print(f"Dashboard: http://localhost:{fe_port}")
        print("Press Ctrl-C to stop everything.")
        print("=" * 60, flush=True)
        while api.poll() is None and (fe is None or fe.poll() is None):
            time.sleep(0.5)
        print("A server stopped unexpectedly; shutting down.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    finally:
        stop(fe)
        stop(api)


if __name__ == "__main__":
    sys.exit(main())
