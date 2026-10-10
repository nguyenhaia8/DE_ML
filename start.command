#!/bin/bash
# Double-click in Finder, or run ./start.command from any working directory.
set -eu

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT_DIR"
export PATH="/opt/homebrew/opt/python@3.13/bin:/usr/local/opt/python@3.13/bin:/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
unset PYTHONHOME PYTHONPATH

fail() {
    printf '\nStartup failed: %s\n' "$1" >&2
    if [ -t 0 ]; then read -r -p 'Press Return to close this window...' || true; fi
    exit 1
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    printf 'Usage: ./start.command [--streamlit] [--no-browser]\n\n'
    printf 'Starts the React frontend and Python API; --streamlit starts the original demo.\n'
    printf 'Keep Terminal open. Press Control-C to stop the services started by this launcher.\n'
    exit 0
fi

# Prefer an existing environment, then Python versions supported by the ML stack.
PYTHON_BIN=""
for candidate in "$PROJECT_DIR/app/.venv/bin/python" python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))' >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    if ! command -v brew >/dev/null 2>&1; then
        fail 'Python 3.10–3.13 is required. Install Python 3.13, then run this file again. If Homebrew is installed, run: brew install python@3.13'
    fi
    printf 'Installing Python 3.13 with Homebrew...\n'
    brew install python@3.13 || fail 'Homebrew could not install Python. Check your internet connection and the Homebrew error above.'
    PYTHON_BIN="$(brew --prefix python@3.13)/bin/python3.13"
fi

if "$PYTHON_BIN" -u "$PROJECT_DIR/app/launcher.py" "$@"; then
    exit 0
else
    launch_status=$?
    if [ -t 0 ]; then read -r -p 'Press Return to close this window...' || true; fi
    exit "$launch_status"
fi
