#!/usr/bin/env python3
"""Download and verify the pinned target and DFlash2 checkpoints on Runpod."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path


DEPLOY = Path(__file__).resolve().parent
MANIFEST = json.loads((DEPLOY / "runpod-b300-manifest.json").read_text())
MODELS = Path("/workspace/models")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and ".cache" not in candidate.relative_to(path).parts
        and candidate.name != "checkpoint-verification.json"
    )


def verify_target(path: Path) -> dict[str, object]:
    expected = MANIFEST["target_model"]
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    referenced_files = sorted(set(index["weight_map"].values()))
    missing = [name for name in referenced_files if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing indexed target files: {missing[:3]}")
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if len(shards) != expected["model_shards"]:
        raise ValueError(
            f"target model shards: expected {expected['model_shards']}, got {len(shards)}"
        )

    files = repository_files(path)
    total_bytes = sum(candidate.stat().st_size for candidate in files)
    if len(files) != expected["file_count"]:
        raise ValueError(
            f"target file count: expected {expected['file_count']}, got {len(files)}"
        )
    if total_bytes != expected["total_bytes"]:
        raise ValueError(
            f"target bytes: expected {expected['total_bytes']}, got {total_bytes}"
        )

    hashes: dict[str, str] = {}
    for name, expected_hash in expected["hashes"].items():
        actual = sha256(path / name)
        if actual != expected_hash:
            raise ValueError(f"target hash mismatch: {name}: {actual}")
        hashes[name] = actual

    return {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository": expected["repository"],
        "revision": expected["revision"],
        "file_count": len(files),
        "model_shards": len(shards),
        "total_bytes": total_bytes,
        "hashes": hashes,
    }


def verify_draft(path: Path) -> dict[str, object]:
    expected = MANIFEST["draft_model"]
    verified_files: dict[str, dict[str, object]] = {}
    for name, file_expected in expected["files"].items():
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        size = candidate.stat().st_size
        if size != file_expected["bytes"]:
            raise ValueError(
                f"draft bytes: {name}: expected {file_expected['bytes']}, got {size}"
            )
        actual = sha256(candidate)
        if actual != file_expected["sha256"]:
            raise ValueError(f"draft hash mismatch: {name}: {actual}")
        verified_files[name] = {"bytes": size, "sha256": actual}

    return {
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository": expected["repository"],
        "revision": expected["revision"],
        "files": verified_files,
    }


def stage(
    *,
    model: dict[str, object],
    destination: Path,
    verify,
    allow_patterns: list[str] | None = None,
) -> None:
    try:
        result = verify(destination)
        print(f"{destination.name}_ALREADY_VERIFIED", flush=True)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        destination.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(model["repository"]),
            revision=str(model["revision"]),
            local_dir=destination,
            allow_patterns=allow_patterns,
            max_workers=8,
        )
        result = verify(destination)
        print(f"{destination.name}_DOWNLOADED", flush=True)

    (destination / "checkpoint-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    target = MANIFEST["target_model"]
    draft = MANIFEST["draft_model"]
    stage(
        model=target,
        destination=Path(str(target["path"])),
        verify=verify_target,
    )
    stage(
        model=draft,
        destination=Path(str(draft["path"])),
        verify=verify_draft,
        allow_patterns=sorted(draft["files"]),
    )
    print("RUNPOD_B300_MODELS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
