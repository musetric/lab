# ruff: noqa: S603
import argparse
import shlex
import subprocess
import sys


def run_commands(commands):
    for cmd in commands:
        cmd_args = shlex.split(cmd) if isinstance(cmd, str) else cmd
        result = subprocess.run(cmd_args, check=False)

        if result.returncode != 0:
            sys.exit(1)


CHECKS = "node node_modules/@musetric/checks/dist/cli.js"


def check():
    commands = [
        "uv lock --check",
        "editorconfig-checker",
        f"{CHECKS} english",
        f"{CHECKS} notices",
        f"{CHECKS} links",
        "uv run --only-group check ruff check . --no-cache",
        "uv run --only-group check isort . --check-only",
        "uv run --only-group check black . --check",
    ]
    run_commands(commands)


def fix():
    commands = [
        "uv run --only-group check ruff check . --fix --no-cache",
        "uv run --only-group check isort .",
        "uv run --only-group check black .",
    ]
    run_commands(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linting checks")
    parser.add_argument("--fix", action="store_true", help="Fix linting issues")
    args = parser.parse_args()

    if args.fix:
        fix()
    else:
        check()
