"""Stream child output into notebook stdout, persist logs, and emit idle heartbeats."""
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path


def run_live(command, *, cwd=None, env=None, timeout=None, log_path=None,
             label="Process", new_session=True):
    started = time.monotonic()
    messages = queue.Queue()
    child_env = dict(os.environ if env is None else env, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=child_env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               start_new_session=new_session and os.name == "posix")
    log = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")

    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    threading.Thread(target=read_output, daemon=True).start()

    def emit(line):
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
        if log:
            log.write(line if line.endswith("\n") else line + "\n")
            log.flush()

    def stop():
        if process.poll() is None:
            if new_session and os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if new_session and os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()

    try:
        emit(f"[START] {label} | PID={process.pid}")
        last_heartbeat = started
        eof = False
        while not eof or process.poll() is None:
            now = time.monotonic()
            if timeout and now-started >= timeout:
                stop()
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                line = messages.get(timeout=1)
                if line is None:
                    eof = True
                else:
                    emit(line)
            except queue.Empty:
                pass
            if now-last_heartbeat >= 30:
                remaining = f" | remaining={max(0, timeout-(now-started))/60:.1f} min" if timeout else ""
                emit(f"[HEARTBEAT] {label} | elapsed={(now-started)/60:.1f} min{remaining}")
                last_heartbeat = now
        result = process.wait()
        emit(f"[END] {label} | exit={result} | elapsed={(time.monotonic()-started)/60:.1f} min")
        if result:
            raise subprocess.CalledProcessError(result, command)
        return result
    except BaseException:
        stop()
        raise
    finally:
        process.stdout.close()
        if log:
            log.close()
