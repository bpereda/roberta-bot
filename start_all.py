import signal
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROCESS_FILES = ("sensor_server.py", "bot.py")
processes: list[subprocess.Popen] = []
stopping = False


def stop_processes(*_args) -> None:
    global stopping
    if stopping:
        return

    stopping = True
    print("Stopping Roberta services...", flush=True)

    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is not None:
            continue

        timeout = max(0, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    for process_file in PROCESS_FILES:
        path = BASE_DIR / process_file
        if not path.exists():
            print(f"Missing required file: {path}", file=sys.stderr)
            return 1

    signal.signal(signal.SIGINT, stop_processes)
    signal.signal(signal.SIGTERM, stop_processes)

    for process_file in PROCESS_FILES:
        print(f"Starting {process_file}...", flush=True)
        processes.append(
            subprocess.Popen(
                [sys.executable, str(BASE_DIR / process_file)],
                cwd=BASE_DIR,
            )
        )

    try:
        while not stopping:
            for process_file, process in zip(PROCESS_FILES, processes):
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"{process_file} exited with status {return_code}",
                        file=sys.stderr,
                        flush=True,
                    )
                    stop_processes()
                    return return_code or 1
            time.sleep(1)
    finally:
        stop_processes()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
