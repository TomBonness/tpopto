#!/usr/bin/env python3
"""Select one isolated DFlash performance experiment in the service env file."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import tempfile
from pathlib import Path

EXPERIMENTS = (
    "baseline",
    "replayssm-spec",
    "topk-v2",
    "kpool-metadata",
    "ingraph-metadata",
    "mhc-post-pre",
)
ENV_NAME = "DFLASH_PERF_EXPERIMENT"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/etc/sglang-glm.env"),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart sglang-glm.service after the atomic env-file update",
    )
    return parser.parse_args()


def replace_experiment(path: Path, experiment: str) -> None:
    current = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{ENV_NAME}="
    updated = [line for line in current if not line.startswith(prefix)]
    updated.append(f"{prefix}{experiment}")

    metadata = path.stat()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write("\n".join(updated) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.chmod(temporary_name, stat.S_IMODE(metadata.st_mode))
        if hasattr(os, "chown"):
            os.chown(temporary_name, metadata.st_uid, metadata.st_gid)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    args = arguments()
    replace_experiment(args.env_file, args.experiment)
    print(f"selected {args.experiment} in {args.env_file}")
    if args.restart:
        subprocess.run(["systemctl", "restart", "sglang-glm.service"], check=True)
        print("restarted sglang-glm.service")


if __name__ == "__main__":
    main()
