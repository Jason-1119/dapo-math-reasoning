"""Install pinned dependencies into a clean venv without modifying Colab's kernel."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def isolated_env():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default="/content/dapo-venv")
    args = parser.parse_args()
    if sys.platform != "linux" or not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Use a Linux Colab runtime with Python 3.10 through 3.13.")
    project = Path(__file__).resolve().parent
    venv = Path(args.venv).resolve()
    python = venv / "bin/python"
    marker = venv / "dapo_environment.json"
    requirements = project / "requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes() + TORCH_INDEX.encode() +
                                 sys.version.encode()).hexdigest()
    env = isolated_env()

    def run(*cmd):
        subprocess.run(list(map(str, cmd)), check=True, env=env, cwd=project)

    if not python.exists():
        # Bootstrap pip from the kernel's pip with --python. Do not inherit site-packages.
        run(sys.executable, "-m", "venv", "--without-pip", venv)
    venv_config = (venv / "pyvenv.cfg").read_text().lower()
    if "include-system-site-packages = false" not in venv_config:
        raise RuntimeError("The target venv inherits system packages. Choose a fresh --venv path.")
    cached = json.loads(marker.read_text()) if marker.exists() else {}
    if cached.get("fingerprint") != fingerprint:
        run(sys.executable, "-m", "pip", "--python", python, "install", "pip==25.2")
        run(python, "-m", "pip", "install", "torch==2.8.0", "--index-url", TORCH_INDEX)
        run(python, "-m", "pip", "install", "-r", requirements)
        run(python, "-m", "pip", "check")
        run(python, "-m", "pip", "freeze", "--all")
        marker.write_text(json.dumps({"fingerprint": fingerprint, "python": str(python)}, indent=2))
    run(python, "-m", "pip", "check")
    run(python, project / "preflight.py")
    print(f"READY: use {python} for every project command. No kernel restart is required.")


if __name__ == "__main__":
    main()
