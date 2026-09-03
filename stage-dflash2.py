#!/usr/bin/env python3
"""Download and verify the pinned GLM-5.3-Flash DFlash2 draft checkpoint."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPOSITORY = "incoai/GLM-5.3-Flash-DFlash2"
REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
DESTINATION = Path("/workspace/models/GLM-5.3-Flash-DFlash2")
FILES = {
    "config.json": {
        "bytes": 1273,
        "sha256": "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573",
    },
    "model.safetensors": {
        "bytes": 2342169800,
        "sha256": "b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict[str, object]:
    verified_files: dict[str, dict[str, object]] = {}
    for name, expected in FILES.items():
        path = DESTINATION / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != expected["bytes"]:
            raise ValueError(f"{name}: expected {expected['bytes']} bytes, got {size}")
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"{name}: expected sha256 {expected['sha256']}, got {actual_hash}"
            )
        verified_files[name] = {"bytes": size, "sha256": actual_hash}

    return {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": verified_files,
    }


def main() -> None:
    try:
        result = verify()
        print("DFLASH2_CHECKPOINT_ALREADY_VERIFIED")
    except (FileNotFoundError, ValueError):
        DESTINATION.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            local_dir=DESTINATION,
            allow_patterns=sorted(FILES),
        )
        result = verify()
        print("DFLASH2_CHECKPOINT_DOWNLOADED")

    verification_path = DESTINATION / "checkpoint-verification.json"
    verification_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("DFLASH2_CHECKPOINT_VERIFIED")


if __name__ == "__main__":
    main()
