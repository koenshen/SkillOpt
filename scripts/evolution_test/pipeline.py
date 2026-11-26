"""Run skillopt_evolution.run with its normal flags, then evaluate its final skill on test."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    python = [sys.executable, "-B", "-u", "-m"]
    command = [*python, "skillopt_evolution.run", *args]
    if "--help" in args or "-h" in args:
        print(__doc__, flush=True)
    if "--inspect" in args or "--help" in args or "-h" in args:
        return subprocess.run(command).returncode

    output = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if output is None and line.startswith("Output: "):
                output = Path(line.removeprefix("Output: ").rstrip("\r\n"))
            elif output is None and line.startswith("Already complete: "):
                # --resume on a finished run reports the summary file.
                summary = Path(line.removeprefix("Already complete: ").rstrip("\r\n"))
                if summary.name == "summary.json":
                    output = summary.parent
        status = process.wait()
    if status:
        return status
    if output is None or not output.is_absolute():
        raise RuntimeError("Evolution reported no absolute output directory; test was not started")
    with (output / "summary.json").open(encoding="utf-8") as stream:
        if json.load(stream).get("status") != "complete":
            raise RuntimeError(f"Evolution has no complete result at {output}; test was not started")
    return subprocess.run([*python, "scripts.evolution_test.run", "--evolution-dir", str(output)]).returncode


if __name__ == "__main__":
    sys.exit(main())
