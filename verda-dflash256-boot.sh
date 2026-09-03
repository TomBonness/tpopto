#!/usr/bin/env bash
set -euo pipefail

DEPLOY=${DEPLOY:-/workspace/deploy}
MODEL=/workspace/models/GLM-5.3-Flash-NVFP4
DRAFT=/workspace/models/GLM-5.3-Flash-DFlash2
MANIFEST=$DEPLOY/verda-dflash256-manifest.json
DFLASH_DRAFT_TOKENS=${DFLASH_DRAFT_TOKENS:-8}

case "$DFLASH_DRAFT_TOKENS" in
  8|16|32|64) ;;
  *) echo "DFLASH_DRAFT_TOKENS must be one of 8, 16, 32, or 64" >&2; exit 2 ;;
esac
: "${SGLANG_API_KEY:?SGLANG_API_KEY must be set}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST=12.0a
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FP8_WO_A_GEMM=0
export XDG_CACHE_HOME=/workspace/cache
export HF_HOME=/workspace/cache/huggingface
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME"

python3 - "$DEPLOY" "$MODEL" "$DRAFT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

deploy, model, draft = map(Path, sys.argv[1:])
base_manifest = json.loads((deploy / "manifest.json").read_text())
profile_manifest = json.loads((deploy / "verda-dflash256-manifest.json").read_text())

for patch in base_manifest["patches"]:
    source = deploy / patch["source"]
    data = source.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != patch["sha256"]:
        raise RuntimeError(f"patch hash mismatch: {source}: {actual}")
    compile(data, str(source), "exec")
    target = Path(patch["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

target_verification = json.loads((model / "checkpoint-verification.json").read_text())
expected_target = profile_manifest["target_model"]
for key in ("file_count", "model_shards", "total_bytes"):
    if target_verification[key] != expected_target[key]:
        raise RuntimeError(
            f"target checkpoint {key}: expected {expected_target[key]}, "
            f"got {target_verification[key]}"
        )

draft_verification = json.loads((draft / "checkpoint-verification.json").read_text())
expected_draft = profile_manifest["draft_model"]
if draft_verification["revision"] != expected_draft["revision"]:
    raise RuntimeError("draft checkpoint revision mismatch")
for name, expected in expected_draft["files"].items():
    actual = draft_verification["files"].get(name)
    if actual != expected:
        raise RuntimeError(f"draft checkpoint verification mismatch: {name}")

print("DFLASH256_RUNTIME_VERIFIED", flush=True)
PY
COMMON=(
  --model-path "$MODEL"
  --served-model-name glm-5.3-flash
  --tp-size 4
  --ep-size 4
  --context-length 262144
  --max-total-tokens 262144
  --max-running-requests 1
  --quantization modelopt_fp4
  --attention-backend dsa
  --dsa-prefill-backend flashinfer_sparse_mla
  --dsa-decode-backend flashinfer_sparse_mla
  --linear-attn-backend triton
  --kv-cache-dtype fp8_e4m3
  --moe-runner-backend flashinfer_cutlass
  --disable-shared-experts-fusion
  --chunked-prefill-size 4096
  --max-prefill-tokens 4096
  --mem-fraction-static 0.85
  --mamba-full-memory-ratio 2
  --cuda-graph-max-bs-decode 1
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT"
  --speculative-num-draft-tokens "$DFLASH_DRAFT_TOKENS"
  --speculative-draft-window-size 2048
  --media-url-max-file-size-mb 1024
  --enable-multimodal
  --chat-template "$DEPLOY/chat-template-mm.jinja"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --api-key "$SGLANG_API_KEY"
  --host 127.0.0.1
  --port 8000
)

echo DFLASH256_PRODUCTION_STARTING
exec python3 -m sglang.launch_server "${COMMON[@]}" \
  >> /workspace/dflash256-production.log 2>&1
