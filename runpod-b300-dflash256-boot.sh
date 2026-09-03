#!/usr/bin/env bash
set -euo pipefail

DEPLOY=${DEPLOY:-/workspace/deploy}
SOURCE_MODEL_ROOT=/workspace/models
LOCAL_MODEL_ROOT=${LOCAL_MODEL_ROOT:-/models-local}
MODEL=$LOCAL_MODEL_ROOT/GLM-5.3-Flash-NVFP4
DRAFT=$LOCAL_MODEL_ROOT/GLM-5.3-Flash-DFlash2

if [[ "$LOCAL_MODEL_ROOT" != "$SOURCE_MODEL_ROOT" ]] && {
  [[ ! -f "$MODEL/checkpoint-verification.json" ]] ||
    [[ ! -f "$DRAFT/checkpoint-verification.json" ]]
}; then
  python3 - "$SOURCE_MODEL_ROOT" "$LOCAL_MODEL_ROOT" <<'PY'
import concurrent.futures
import shutil
import sys
import time
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
files = [
    path
    for path in source.rglob("*")
    if path.is_file() and ".cache" not in path.relative_to(source).parts
]
for path in files:
    (destination / path.relative_to(source)).parent.mkdir(
        parents=True, exist_ok=True
    )

started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    list(
        executor.map(
            lambda path: shutil.copy2(
                path, destination / path.relative_to(source)
            ),
            files,
        )
    )
print(
    "RUNPOD_B300_LOCAL_MODEL_CACHE_READY "
    f"files={len(files)} bytes={sum(path.stat().st_size for path in files)} "
    f"seconds={time.monotonic() - started:.2f}"
)
PY
fi
MANIFEST=$DEPLOY/runpod-b300-manifest.json
DFLASH_DRAFT_TOKENS=${DFLASH_DRAFT_TOKENS:-8}
DFLASH_PERF_EXPERIMENT=${DFLASH_PERF_EXPERIMENT:-baseline}

case "$DFLASH_DRAFT_TOKENS" in
  8|16|32|64) ;;
  *) echo "DFLASH_DRAFT_TOKENS must be one of 8, 16, 32, or 64" >&2; exit 2 ;;
esac

EXPERIMENT_ARGS=()
DSA_PREFILL_BACKEND=tilelang
DSA_DECODE_BACKEND=tilelang
DSA_PAGED_MQA_LOGITS_BACKEND=deepgemm
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_DSA_FUSE_TOPK=1
export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=0
export SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=0
export SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA_DG_OUT_OF_GRAPH=0
export SGLANG_GLM53_FUSE_MHC_POST_PRE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
case "$DFLASH_PERF_EXPERIMENT" in
  baseline) ;;
  replayssm-spec)
    EXPERIMENT_ARGS+=(--enable-linear-replayssm-spec)
    ;;
  topk-v2)
    export SGLANG_OPT_USE_TOPK_V2=1
    ;;
  kpool-metadata)
    export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1
    ;;
  ingraph-metadata)
    export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1
    export SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1
    ;;
  mhc-post-pre)
    export SGLANG_GLM53_FUSE_MHC_POST_PRE=1
    ;;
  deepgemm-hc-prenorm)
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
    ;;
  cutedsl-paged-mqa)
    DSA_PAGED_MQA_LOGITS_BACKEND=cutedsl
    ;;
  blackwell-dsa)
    DSA_PREFILL_BACKEND=flashmla_sparse
    DSA_DECODE_BACKEND=trtllm
    ;;
  b300-cyclepack)
    EXPERIMENT_ARGS+=(--enable-linear-replayssm-spec)
    DSA_PAGED_MQA_LOGITS_BACKEND=cutedsl
    export SGLANG_OPT_USE_TOPK_V2=1
    export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1
    export SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1
    export SGLANG_GLM53_FUSE_MHC_POST_PRE=1
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
    ;;
  *)
    echo "DFLASH_PERF_EXPERIMENT must be baseline, replayssm-spec, topk-v2, kpool-metadata, ingraph-metadata, mhc-post-pre, deepgemm-hc-prenorm, cutedsl-paged-mqa, blackwell-dsa, or b300-cyclepack" >&2
    exit 2
    ;;
esac
: "${SGLANG_API_KEY:?SGLANG_API_KEY must be set}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST=10.3a
export FLASHINFER_CUDA_ARCH_LIST=10.3a
export SGLANG_OPT_USE_TILELANG_INDEXER=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_OPT_FP8_WO_A_GEMM=1
export XDG_CACHE_HOME=/workspace/cache
export HF_HOME=/workspace/cache/huggingface
export SGLANG_CACHE_DIR=/workspace/cache/sglang
export TRITON_CACHE_DIR=$SGLANG_CACHE_DIR/triton
export TORCHINDUCTOR_CACHE_DIR=$SGLANG_CACHE_DIR/torchinductor
export CUDA_CACHE_PATH=$SGLANG_CACHE_DIR/cuda
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$SGLANG_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH"

python3 - "$DEPLOY" "$MODEL" "$DRAFT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

deploy, model, draft = map(Path, sys.argv[1:])
base_manifest = json.loads((deploy / "manifest.json").read_text())
profile_manifest = json.loads((deploy / "runpod-b300-manifest.json").read_text())

for patch in base_manifest["patches"]:
    source = deploy / patch["source"]
    data = source.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != patch["sha256"]:
        raise RuntimeError(f"patch hash mismatch: {source}: {actual}")
    if source.suffix == ".py":
        compile(data, str(source), "exec")
    target = Path(patch["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

target_verification = json.loads((model / "checkpoint-verification.json").read_text())
expected_target = profile_manifest["target_model"]
if target_verification["revision"] != expected_target["revision"]:
    raise RuntimeError("target checkpoint revision mismatch")
for key in ("file_count", "model_shards", "total_bytes", "hashes"):
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
    if draft_verification["files"].get(name) != expected:
        raise RuntimeError(f"draft checkpoint verification mismatch: {name}")

print("RUNPOD_B300_RUNTIME_VERIFIED", flush=True)
PY

COMMON=(
  --model-path "$MODEL"
  --served-model-name glm-5.3-flash
  --tp-size 1
  --ep-size 1
  --context-length 131072
  --max-total-tokens 131072
  --max-running-requests 1
  --quantization modelopt_fp4
  --attention-backend dsa
  --dsa-prefill-backend "$DSA_PREFILL_BACKEND"
  --dsa-decode-backend "$DSA_DECODE_BACKEND"
  --dsa-paged-mqa-logits-backend "$DSA_PAGED_MQA_LOGITS_BACKEND"
  --linear-attn-backend triton
  --kv-cache-dtype bfloat16
  --moe-runner-backend flashinfer_cutedsl
  --disable-shared-experts-fusion
  --chunked-prefill-size 4096
  --max-prefill-tokens 4096
  --mem-fraction-static 0.88
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
  --host 0.0.0.0
  --port 8000
)

LOG=/workspace/dflash256-b300-${DFLASH_PERF_EXPERIMENT}.log
echo "RUNPOD_B300_DFLASH256_STARTING experiment=$DFLASH_PERF_EXPERIMENT draft_tokens=$DFLASH_DRAFT_TOKENS"
exec python3 -m sglang.launch_server "${COMMON[@]}" "${EXPERIMENT_ARGS[@]}" >>"$LOG" 2>&1
