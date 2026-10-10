"""macOS launcher; only the Python standard library is needed to bootstrap."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "app" / ".venv"
FRONTEND = ROOT / "app" / "frontend"
HTTP = build_opener(ProxyHandler({}))  # Local servers should bypass proxy settings.
SAMPLE = "An experienced developer who builds reliable products, collaborates with colleagues, plans projects carefully, learns new tools, mentors teams, and communicates clearly with customers."


class StartupError(Exception):
    pass


class Launcher:
    def __init__(self, args):
        self.args = args
        self.runtime = ROOT / ".run" / ("streamlit" if args.streamlit else "web")
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.children: list[subprocess.Popen] = []
        self.env = os.environ.copy()
        self.env.pop("PYTHONHOME", None)
        self.env.pop("PYTHONPATH", None)
        self.env.update(PYTHONUNBUFFERED="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        self.python = str(VENV / "bin" / "python")

    def run(self, command, *, cwd=ROOT, log="setup.log", capture=False):
        with (self.runtime / log).open("ab") as output:
            try:
                process = subprocess.Popen(
                    [str(part) for part in command], cwd=cwd, env=self.env,
                    stdout=subprocess.PIPE if capture else output,
                    stderr=output, start_new_session=True,
                )
            except OSError as exc:
                raise StartupError(f"Could not run {command[0]}: {exc}") from exc
            self.children.append(process)
            result, _ = process.communicate()
            self.children.remove(process)
        if process.returncode:
            raise StartupError(f"Command failed: {' '.join(map(str, command))}. See {self.runtime / log}")
        return result.decode().strip() if capture else ""

    def probe(self, command, *, cwd=ROOT):
        try:
            self.run(command, cwd=cwd)
            return True
        except StartupError:
            return False

    def ensure_venv(self):
        valid = self.probe([self.python, "-c", "import sys; from pathlib import Path; assert sys.prefix != sys.base_prefix; assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(); assert (3, 10) <= sys.version_info[:2] <= (3, 13)", VENV]) if (VENV / "bin" / "python").exists() else False
        if not valid:
            if VENV.exists() or VENV.is_symlink():
                backup = VENV.with_name(f".venv.backup-{time.time_ns()}")
                VENV.rename(backup)
                print(f"Preserved unusable environment at {backup}")
            print("Creating the project Python environment...")
            base_python = getattr(sys, "_base_executable", sys.executable)
            self.run([base_python, "-m", "venv", VENV])

        if not self.probe([self.python, "-m", "pip", "--version"]):
            self.run([self.python, "-m", "ensurepip", "--upgrade"])

    def ensure_python(self):
        self.ensure_venv()
        requirements = [ROOT / "app/backend/requirements.txt", ROOT / "source/requirements.txt"]
        check = """
import importlib.metadata as metadata
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
import sys
for filename in sys.argv[1:]:
    for line in Path(filename).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        req = Requirement(line)
        if req.marker is None or req.marker.evaluate():
            assert metadata.version(req.name) in req.specifier, str(req)
"""
        if not self.probe([self.python, "-c", check, *requirements]):
            print("Installing missing or outdated Python dependencies (first setup can take several minutes)...")
            self.run([self.python, "-m", "pip", "install", "--retries", "2", "--timeout", "30", "-r", requirements[0], "-r", requirements[1]])

        imports = "import fastapi, uvicorn, multipart, joblib, sklearn, pdfplumber, pandas, streamlit, plotly, xgboost, sentence_transformers"
        if not self.probe([self.python, "-c", imports]):
            # macOS XGBoost wheels can require the Homebrew OpenMP runtime.
            setup_log = (self.runtime / "setup.log").read_text(errors="replace")
            if "libomp" in setup_log and shutil.which("brew"):
                print("Installing the OpenMP runtime required by XGBoost...")
                self.run(["brew", "install", "libomp"])
            if not self.probe([self.python, "-c", imports]):
                print("Repairing the Python packages...")
                self.run([self.python, "-m", "pip", "install", "--force-reinstall", "--retries", "2", "--timeout", "30", "-r", requirements[0], "-r", requirements[1]])
                self.run([self.python, "-c", imports])
        if not self.probe([self.python, "-m", "pip", "check"]):
            print("Repairing inconsistent Python dependencies...")
            self.run([self.python, "-m", "pip", "install", "--upgrade", "--retries", "2", "--timeout", "30", "-r", requirements[0], "-r", requirements[1]])
            self.run([self.python, "-m", "pip", "check"])

    def ensure_node(self):
        candidates = [shutil.which("node")]
        candidates += [str(path) for pattern in (".nvm/versions/node/*/bin/node", ".local/share/fnm/node-versions/*/installation/bin/node", ".volta/bin/node") for path in sorted(Path.home().glob(pattern), reverse=True)]
        candidates += ["/opt/homebrew/opt/node/bin/node", "/usr/local/opt/node/bin/node"]
        for attempt in range(2):
            for candidate in candidates:
                if not candidate or not Path(candidate).is_file():
                    continue
                try:
                    version = self.run([candidate, "--version"], capture=True)
                    major, minor, _ = map(int, version.lstrip("v").split(".")[:3])
                    if not ((major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major > 22):
                        continue
                    self.env["PATH"] = str(Path(candidate).parent) + os.pathsep + self.env["PATH"]
                    npm = shutil.which("npm", path=self.env["PATH"])
                    if npm and self.probe([npm, "--version"]):
                        print(f"Using Node {version}")
                        return npm
                except (ValueError, StartupError):
                    continue
            if attempt == 0 and shutil.which("brew"):
                print("Installing a compatible Node.js with Homebrew...")
                self.run(["brew", "install", "node"])
                prefix = self.run(["brew", "--prefix", "node"], capture=True)
                candidates.insert(0, str(Path(prefix) / "bin/node"))
        raise StartupError("Node.js 20.19+ or 22.12+ and npm are required. Install a current Node.js release and run start.command again.")

    def ensure_frontend(self):
        npm = self.ensure_node()
        stamp = self.runtime / "frontend-dependencies.json"
        fingerprint = json.dumps({
            "package": (FRONTEND / "package.json").read_text(),
            "lock": (FRONTEND / "package-lock.json").read_text() if (FRONTEND / "package-lock.json").exists() else None,
            "node": self.run(["node", "--version"], capture=True),
            "root": str(ROOT),
        }, sort_keys=True)
        installed = self.probe([npm, "ls", "--depth=0"], cwd=FRONTEND)
        working = installed and self.probe(["node", "--input-type=module", "-e", "await import('vite'); await import('@vitejs/plugin-react'); await import('react'); await import('lucide-react'); await import('typescript')"], cwd=FRONTEND)
        if not working or (stamp.exists() and stamp.read_text() != fingerprint):
            print("Installing or repairing frontend dependencies...")
            self.env.pop("NODE_ENV", None)
            if (FRONTEND / "package-lock.json").exists() and self.probe([npm, "ci", "--include=dev", "--no-audit", "--no-fund"], cwd=FRONTEND):
                pass
            else:
                self.run([npm, "install", "--include=dev", "--no-audit", "--no-fund"], cwd=FRONTEND)
            self.run(["node", "--input-type=module", "-e", "await import('vite'); await import('@vitejs/plugin-react')"], cwd=FRONTEND)
            # npm install can update a stale lockfile; record the resulting inputs.
            inputs = json.loads(fingerprint)
            inputs["lock"] = (FRONTEND / "package-lock.json").read_text()
            fingerprint = json.dumps(inputs, sort_keys=True)
        stamp.write_text(fingerprint)

    def ensure_baseline(self):
        model = ROOT / "source/models/baseline.joblib"
        check = "import joblib, sys; b=joblib.load(sys.argv[1]); assert b['model_type'] == 'baseline_tfidf_svd_rf'; x=b['svd'].transform(b['vectorizer'].transform([sys.argv[2]])); assert len(b['trait_cols']) == 5; [b['classifiers'][t].predict_proba(x) for t in b['trait_cols']]"
        if model.exists() and self.probe([self.python, "-c", check, model, SAMPLE]):
            return
        print("Baseline model is missing or unusable. Training a replacement; this can take several minutes...")
        essays = ROOT / "source/essays.csv"
        if not essays.exists():
            print("Downloading the labeled essay dataset...")
            temporary = self.runtime / "essays.csv"
            with urlopen("https://raw.githubusercontent.com/SenticNet/personality-detection/master/essays.csv", timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(essays)
        replacement = self.runtime / "baseline.joblib"
        self.run([self.python, ROOT / "source/train_baseline.py", "--data", essays, "--out", replacement], log="training.log")
        self.run([self.python, "-c", check, replacement, SAMPLE], log="training.log")
        model.parent.mkdir(parents=True, exist_ok=True)
        if model.exists():
            model.rename(model.with_name(f"baseline.backup-{time.time_ns()}.joblib"))
        replacement.replace(model)

    def start(self, command, *, cwd, log):
        with (self.runtime / log).open("wb") as output:
            child = subprocess.Popen(command, cwd=cwd, env=self.env, stdout=output, stderr=output, start_new_session=True)
        self.children.append(child)
        return child

    def wait_ready(self, child, url, *, backend=False, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise StartupError(f"A server exited during startup (status {child.returncode}).")
            try:
                with HTTP.open(url, timeout=1) as response:
                    payload = response.read()
                    if response.status == 200 and (not backend or json.loads(payload).get("ok") is True):
                        return
            except (OSError, ValueError, URLError):
                pass
            time.sleep(0.2)
        raise StartupError(f"Timed out waiting for {url}")

    def serve(self):
        for attempt in range(3):
            backend_port = free_port(8000)
            frontend_port = free_port(8501 if self.args.streamlit else 5173, exclude=backend_port)
            backend_url = f"http://127.0.0.1:{backend_port}"
            frontend_url = f"http://127.0.0.1:{frontend_port}"
            try:
                if self.args.streamlit:
                    ui = self.start([self.python, "-m", "streamlit", "run", "app.py", "--server.address", "127.0.0.1", "--server.port", str(frontend_port), "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT / "source", log="frontend.log")
                    self.wait_ready(ui, frontend_url + "/_stcore/health")
                else:
                    self.env["CV_PERSONALITY_ALLOWED_ORIGINS"] = frontend_url + f",http://localhost:{frontend_port}"
                    api = self.start([self.python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(backend_port)], cwd=ROOT / "app", log="backend.log")
                    self.wait_ready(api, backend_url + "/api/v1/health/ready", backend=True)
                    # Confirm that the API can actually score a CV before opening the UI.
                    request = Request(backend_url + "/api/v1/predict", data=json.dumps({"text": SAMPLE, "model_id": "baseline"}).encode(), headers={"Content-Type": "application/json"})
                    with HTTP.open(request, timeout=30) as response:
                        assert len(json.load(response)["scores"]) == 5
                    self.env["VITE_API_BASE_URL"] = backend_url
                    ui = self.start(["node", "node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", str(frontend_port), "--strictPort"], cwd=FRONTEND, log="frontend.log")
                    self.wait_ready(ui, frontend_url)
                break
            except StartupError:
                # Retry a port-selection race; report other startup failures immediately.
                collision = any("address already in use" in path.read_text(errors="replace").lower() or "is already in use" in path.read_text(errors="replace").lower() for path in self.runtime.glob("*.log") if path.name != "setup.log")
                self.stop()
                if attempt == 2 or not collision:
                    raise
        state = {"url": frontend_url, "pid": os.getpid(), "ready": True}
        temporary = self.runtime / "state.tmp"
        temporary.write_text(json.dumps(state))
        temporary.replace(self.runtime / "state.json")
        print(f"\nProject is ready: {frontend_url}")
        if not self.args.streamlit:
            print(f"API documentation: {backend_url}/docs")
        print(f"Logs: {self.runtime}\nKeep this window open. Press Control-C to stop.")
        if not self.args.no_browser:
            webbrowser.open(frontend_url)
        while True:
            for child in self.children:
                if child.poll() is not None:
                    raise StartupError(f"A server stopped unexpectedly (status {child.returncode}). Run start.command again to restart.")
            time.sleep(0.5)

    def stop(self):
        for child in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in self.children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        self.children.clear()


def free_port(preferred, *, exclude=None):
    for port in range(preferred, preferred + 100):
        if port == exclude:
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            if port != exclude:
                return port


def request_stop(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description="Set up and start the CV personality project on macOS.")
    parser.add_argument("--streamlit", action="store_true", help="start the original Streamlit demo instead of React + FastAPI")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    args = parser.parse_args()
    launcher = Launcher(args)
    with (launcher.runtime / "launcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("This project is already starting or running in another Terminal window.")
            try:
                state = json.loads((launcher.runtime / "state.json").read_text())
                if state.get("ready"):
                    print(f"Project URL: {state['url']}")
                    if not args.no_browser:
                        webbrowser.open(state["url"])
            except (OSError, ValueError):
                pass
            return 0
        (launcher.runtime / "state.json").unlink(missing_ok=True)
        # Serialize setup across the web and Streamlit modes (shared environments).
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGHUP, request_stop)
        try:
            print(f"Starting CV Personality Prediction from {ROOT}")
            with (ROOT / ".run/setup.lock").open("w") as setup_lock:
                try:
                    fcntl.flock(setup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("Waiting for the other launcher to finish dependency setup...")
                    fcntl.flock(setup_lock, fcntl.LOCK_EX)
                (launcher.runtime / "setup.log").write_text("")
                launcher.ensure_python()
                if not args.streamlit:
                    launcher.ensure_frontend()
                launcher.ensure_baseline()
            launcher.serve()
        except KeyboardInterrupt:
            print("\nStopping the project...")
            return 0
        except (StartupError, OSError, ValueError, AssertionError) as exc:
            print(f"\nStartup failed: {exc}", file=sys.stderr)
            for filename in ("setup.log", "training.log", "backend.log", "frontend.log"):
                path = launcher.runtime / filename
                if path.exists() and path.stat().st_size:
                    print(f"\nLast lines of {path}:\n" + "\n".join(path.read_text(errors="replace").splitlines()[-12:]), file=sys.stderr)
            return 1
        finally:
            launcher.stop()
            (launcher.runtime / "state.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
