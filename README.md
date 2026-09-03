# Exact GLM-5.3-Flash-NVFP4 SGLang deployment bundle

Locked deployment sources for the validated 4× RTX PRO 6000 Blackwell workstation profile and the retained single-B300 Runpod profile. Both use the exact `lmsysorg/sglang:glm-5.3-flash` image digest; each profile preserves its own server arguments, hardware-specific kernel controls, and checkpoint revisions.

The 182 GiB checkpoint is intentionally **not** copied into the image. Compose mounts the verified local checkpoint read-only from `/mnt/llm_models/GLM-5.3-Flash-NVFP4` and reuses the writable SGLang cache at `/home/ser/.cache/sglang-glm53`. The eight compatibility/performance patches are **baked into the image** at their editable-install paths, so no bind-mounted code is needed.

## Lock points

- Runtime image: `lmsysorg/sglang:glm-5.3-flash@sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290`
- Model: `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `9e0d74e3cef17f634e84fb8e2223707e02616290` (120 shards)
- Server model ID: `glm-5.3-flash` (`owned_by: sglang`)
- Context: 1,048,576 tokens; KV cache `fp8_e4m3`, 3,789,184 tokens per rank (~28 GB)
- TP/EP: 4 / 4
- MoE runner: **`flashinfer_cutlass`** — SM120-native CUTLASS fp4 MoE
- Attention: DSA backend with `flashinfer_sparse_mla` prefill/decode; linear attention via `triton` (KDA)
- Speculative decode: NEXTN (MTP), 5 steps, topk 1, 6 draft tokens
- Parsers: reasoning `glm45`, tool-call `glm47` — exactly what the model card prescribes
- CUDA graphs: full decode capture for batch sizes ≤ 8

## Why not flashinfer_cutedsl

FlashInfer's CuteDSL MoE stack hard-codes `supported_major_versions=[10]` (datacenter Blackwell sm100/sm103 tcgen05 kernels) in `flashinfer/jit/moe_utils.py::gen_moe_utils_module`. On consumer/pro workstation Blackwell (SM120, major version 12) startup dies during autotuning with:

```
RuntimeError: No supported CUDA architectures found for major versions [10].
```

No environment variable can fix this — the caller excludes major 12 by name. `flashinfer_cutlass` ships a dedicated `gen_cutlass_fused_moe_sm120_module` in this image and is validated here.

## Build and run

```bash
docker compose build --pull=false
docker compose up -d
curl http://127.0.0.1:8000/health            # container listens on 30000
```

Startup on this engine takes several minutes (182 GiB load + JIT + graph capture). `GET /health` returns 200 when ready; generation tests then show ~143 tok/s at low reasoning effort and ~208 tok/s at default effort on a single stream (NEXTN drafting).

## Client usage

```python
openai.chat.completions.create(
    model="glm-5.3-flash",
    messages=[{"role": "user", "content": "..."}],
    extra_body={
        "chat_template_kwargs": {
            "reasoning_effort": "low",   # 'low' | 'high'; unset => 'max'
            # "clear_thinking": True,
        }
    },
)
```

- Do **not** pass small `max_tokens`: under default (max) thinking effort the visible `content` field stays empty while the budget burns inside `<think>`. Omit it and let the model stop naturally.
- Sampling per the checkpoint's `generation_config.json`: temperature 1.0, top_p 0.95.
- Tool calls work by sending `tools` + `tool_choice` in the request; sglang needs no enable flag for this.

## Provenance

`manifest.json` records the base digest, model revision, hashes of the four core checkpoint metadata files, and sha256 of every baked patch. It is a configuration lock, not a replacement for validating all 120 model shards after transfer.

## Verda DFlash2 256K profile

The width-eight profile has been verified end to end on 4× RTX PRO 6000 SM120. One useful coding request (45 prompt tokens, 125 completion tokens) measured 981.2 ms TTFT and 464.17 output tokens/s. No synthetic attractor or width sweep was run.

Lock points:

- Target checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `caca4e6a4ebbd66f159d3d2fc256683fd6e27177`
- Draft checkpoint: `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a` (2,342,169,800-byte weights)
- Generic DFlash implementation: SGLang `033446bb05f35c0943aed2750c443077ffc0b92c`
- GLM hidden-state adapter: upstream PR `#36708`, ported into the bundled model patch with the reported `residual is None` guard
- Context/token pool: 262,144 / 262,144
- TP/EP: 4 / 4
- Concurrency: one request
- DFlash2 block: eight draft tokens by default; the launcher accepts only the explicit 8, 16, 32, or 64 candidates through `DFLASH_DRAFT_TOKENS`
- Prefill chunk: 4,096; static memory fraction: 0.85

The bundled GLM patch restores the upstream fused KDA projection path when ModelOpt explicitly excludes both fused projection names from quantization, as this checkpoint does. Each of the 34 KDA layers now uses one merged BF16 projection and one batched BF16 projection instead of six separate projection calls: 136 fewer GEMM calls per target-model pass. The fused path loaded on SM120 and produced a coherent Python LRU-cache implementation.

Draft width remains a separate startup profile; DFlash does not support adaptive width. The live comparison selected width 8: it decoded at 428.56 tokens/s with a 6.8 average accepted length, while width 32 decoded at 314.70 tokens/s and did not expose acceptance telemetry after its single request. Width 32 was 26.57% slower, so `/etc/sglang-glm.env`, the launcher default, and the manifest all retain `DFLASH_DRAFT_TOKENS=8`.

Completed live trial (2026-09-03 UTC):

1. All four RTX PRO 6000 GPUs reported `PHB` topology with no NVLink, and every directed CUDA peer-read check passed. SGLang's TP4 PCIe custom-all-reduce guard remains unchanged; NCCL is used.
2. Width 8 loaded the fused KDA projections and produced coherent output at 428.56 decode tokens/s with 118.69 ms TTFT.
3. Width 32 produced coherent output at 314.70 decode tokens/s with 439.76 ms TTFT. It failed the required 5% gain by a wide margin.
4. Width 8 remains the production profile. Widths 16 and 64 were not run because width 32 lost decisively.
5. Profile the width-eight configuration around one useful request only if more kernel work is justified. The next profile should decide among TP4 collectives, KDA recurrent geometry, mHC post/pre fusion, and the NoPE DSA bridge; do not change several paths in one run.

### Ranked next kernel experiments

The production launcher stays unchanged until an isolated candidate wins. All
measurements use the existing deterministic width-eight request, normalize
kernel time by accepted token, and record decode tokens/s, TTFT, average
acceptance, peak memory, and the named kernel family's GPU time. Keep a change
only when output remains coherent, acceptance does not materially regress, the
targeted kernel time falls, and end-to-end decode improves by at least 3%.

Risk-adjusted run order:

1. **Capture one width-eight kernel timeline.** Attribute target-verify time to
   KDA convolution/recurrent updates, mHC pre/post kernels, DSA top-k and
   metadata, the NoPE sparse-MLA bridge, MoE, and NCCL. This is a measurement
   run, not another draft-width sweep.
2. **Enable speculative ReplaySSM alone** with
   `--enable-linear-replayssm-spec`. The pinned SGLang path explicitly supports
   DFLASH, a linear top-k-one chain, and Triton KDA verification. It stores raw
   `(v, k, g, beta)` history and folds only the accepted prefix instead of
   snapshotting the full recurrent state for every draft token. For 34 local
   KDA states, 16 heads, and a 128x128 state, eight snapshots represent an
   estimated 136-272 MiB per target pass and rank at BF16-FP32. Record the
   automatically selected FP32 state dtype and resulting memory headroom.
3. **Enable `SGLANG_OPT_USE_TOPK_V2=1` alone.** The pinned decode-shaped v2 path
   accepts the exact top-k 2048 geometry and fuses selection with the physical
   page-table transform, avoiding the page-size-one table on that path. The
   bundle installs the exact corrected header from SGLang PR `#37625`
   (`1436d19a9a55cb9fa0ea93e2f266a0c41ef25d0f`); the older kernel could discard
   valid candidates when a radix bin exceeded its 4,096-entry stash.
4. **Enable KPool metadata fusion alone** with
   `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1`. The bundled backend's
   validated envelope exactly matches index-kpool 4, page size 64, and top-k
   2048. If it wins, separately enable
   `SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1`; KPool fusion is required
   before in-graph verify metadata is eligible for this model.

The bundle now exposes the safe-to-launch candidates through one selector,
`DFLASH_PERF_EXPERIMENT`. Allowed values are `baseline`, `replayssm-spec`,
`topk-v2`, `kpool-metadata`, `ingraph-metadata`, and `mhc-post-pre`. The last
candidate wires SGLang's existing `mhc_fused_post_pre` helper into GLM's
attention-to-MLP boundary; it is isolated behind an environment gate and the
baseline retains the original separate post/pre calls. The in-graph
metadata candidate enables its required KPool fusion in the same profile.

On the GPU VM, select and restart one candidate atomically:

```bash
python3 /workspace/deploy/select-dflash-experiment.py baseline --restart
# After the baseline result, replace baseline with exactly one candidate above.
```

Each restart writes its server output to
`/workspace/dflash256-<experiment>.log`. After `/health` returns 200, source
`/etc/sglang-glm.env` and run `benchmark-minimal.py` once. Its JSON records
`dflash_perf_experiment`, active draft width, acceptance, TTFT, and decode
throughput, preventing results from being attributed to the wrong candidate.
By default each result is retained as
`/workspace/dflash256-<experiment>-benchmark.json`.

Profile-gated implementation queue:

1. **Native SM120 NoPE sparse MLA.** Specialize the existing FlashInfer kernel
   for H512 queries, 528-byte FP8 cache rows, zero RoPE, and top-k 2052. The
   current compatibility bridge pads the four-entry tail to 128, materializes
   2,176 H576 rows per query, launches attention twice, and merges two outputs
   through LSE. At eight verify rows across 11 full-attention layers, the
   temporary compact-cache writes alone are about 120 MiB per pass and rank.
   A native specialization removes the Q padding, compaction, second attention
   launch, and pointwise merge.
2. **Fuse DFlash capture contraction into its producer.** The usual mHC
   capture path reaches `hc_contract` with `residual=None`, so replacing only
   the guarded add is not enough. If the trace shows this reduction, have the
   preceding mHC post emit the H-sized average directly at the five capture
   points; preserve the current guarded fallback for non-mHC paths.
3. **Fuse the width-eight KDA verify chain.** A specialized kernel can combine
   the causal convolution update, recurrent KDA verify update, and gated
   RMSNorm while retaining the exact `lower_bound=-5.0` math. The current chain
   materializes its intermediate tensors and uses three launches in each of 34
   KDA layers; a one-launch path can remove up to 68 launches per target pass.
4. **Microbenchmark TP4 collectives before changing them.** The machine is PHB
   PCIe without NVLink, so SGLang deliberately leaves its greater-than-two-GPU
   custom all-reduce disabled. Compare NCCL against one-shot peer all-reduce at
   the actual width-eight H4096 tensors only if the timeline puts collectives
   among the two largest costs; never force the override from topology alone.

Two tempting changes are intentionally below this queue. FlashInfer MoE fused
finalization is already enabled by default, while its CuteDSL MoE generator
excludes SM120. Removing KDA's stale `lower_bound is None` packed-decode guard
is a valid non-speculative decode optimization because the Triton packed kernel
already accepts `lower_bound`, but DFlash target verification uses the separate
`target_verify` path, so it does not attack the measured steady-state hot path.

Prepared files:

- `verda-dflash256-manifest.json` — revisions, hashes, retained Verda resource IDs, and launch contract
- `stage-dflash2.py` — resumable pinned download with full SHA-256 verification
- `stage-dflash2.sh` — runs the stager using the already-cached SGLang image and refuses an implicit large image pull
- `verda-dflash256-boot.sh` — validates both checkpoints and every runtime patch before starting SGLang
- `patches/sglang-communicator_mhc-glm53.py` — opt-in mHC attention-post/FFN-pre fusion dispatch
- `select-dflash-experiment.py` — atomic candidate selection with optional service restart
- `sglang-glm-dflash256.service` — replacement systemd unit
- `benchmark-minimal.py` — exactly one useful coding request, capped at 256 completion tokens, plus candidate identity, active width, and optional average acceptance
- `patches/sglang-kpool_topk_transform-glm53.cuh` — exact PR `#37625` top-k v2 correctness kernel, hash-locked in `manifest.json`

To avoid billing four GPUs during preparation, first boot the retained OS volume on Verda's `CPU.4V.16G` type, attach the retained model volume, copy this bundle to `/workspace/deploy`, run `stage-dflash2.sh`, install the prepared systemd unit, and delete the CPU VM without `--with-volumes`. The exact location, instance types, volume IDs, and SSH key ID are locked in `verda-dflash256-manifest.json`.

After recreating the four-GPU VM from those two volumes:

```bash
install -m 0644 /workspace/deploy/sglang-glm-dflash256.service /etc/systemd/system/sglang-glm.service
systemctl daemon-reload
systemctl enable --now sglang-glm.service
```

Run only the single prepared benchmark after `/health` becomes ready:

```bash
set -a
. /etc/sglang-glm.env
set +a
python3 /workspace/deploy/benchmark-minimal.py
```

The benchmark does not restart the server for a baseline, generate a repetitive attractor, sweep draft widths, or synthesize a long prompt. Measure only one explicitly selected candidate per restart.

## Runpod single-B300 profile

`runpod-b300-manifest.json` locks the resumable TP1 profile for an NVIDIA B300
SXM6 AC (`sm_103a`) in Runpod `EUR-IS-1`. The target and DFlash2 checkpoints,
verification records, compiled-kernel cache, and deployment bundle live on
network volume `1u0a27qypv`; a B300 pod is not required while the deployment is
parked.
The optimized bundle was staged at `2026-09-03T05:57:37Z`; the staging run
revalidated the target shard inventory and locked metadata, fully hashed the
2.34 GB draft checkpoint, and left no B300 pod active. Its per-file deployment
record is `/workspace/runpod-b300-stage-verification.json`.

The prior live width-eight coding request produced coherent output at 436.47
decode tokens/s with 1,137.18 ms TTFT. That number records the pre-optimization
run, not the unlaunched candidates below. TP1 removes the TP4 NCCL path and the
SM120 NoPE bridge from the hot path, leaving KDA/mHC, DSA indexer/top-k/metadata,
sparse attention, and MoE as the relevant kernel surfaces.

The B300 baseline keeps every invasive fusion disabled, but restores the pinned
SGLang SM10x defaults that the earlier launcher inherited from the SM120
workaround: non-TileLang DSV4 indexer selection, non-Torch paged-MQA fallback,
and FP8 `wo_a` GEMM. These switches are explicit so a reused pod environment
cannot silently change the baseline. `SGLANG_OPT_DEEPGEMM_HC_PRENORM` remains
off in the baseline and is measured separately.

Select exactly one isolated candidate with `DFLASH_PERF_EXPERIMENT`:

- `replayssm-spec` — fold only the accepted recurrent-state prefix.
- `topk-v2` — corrected JIT radix selection plus direct plan output.
- `kpool-metadata` — fuse the KPool metadata transforms.
- `ingraph-metadata` — add the target-verify metadata refresh to the CUDA graph.
- `mhc-post-pre` — fuse the attention-to-MLP mHC boundary.
- `deepgemm-hc-prenorm` — replace the split-k mHC prenorm path with the SM10x
  DeepGEMM kernel.
- `cutedsl-paged-mqa` — use the SM10x CuTe DSL indexer kernel; the pinned SGLang
  interface identifies it as the low-batch, long-context path.
- `blackwell-dsa` — compare SGLang's Blackwell BF16 defaults
  (`flashmla_sparse` prefill, `trtllm` decode) against the validated TileLang
  backend.

`b300-cyclepack` combines every candidate above except `blackwell-dsa`. It is an
integration experiment, is **not** production-ready, and is never selected by
default. Promote it only after its members produce coherent output in isolation,
do not materially reduce DFlash acceptance, reduce the targeted kernel time,
and improve repeated end-to-end decode throughput by at least 3%.

Prepared Runpod files:

- `runpod-b300-manifest.json` — image/checkpoint locks, prior live measurement,
  persistent-volume contract, and candidate definitions
- `stage-runpod-b300.py` — resumable checkpoint download and verification
- `runpod-b300-dflash256-boot.sh` — parallel local-NVMe model copy, patch/hash
  verification, persistent JIT caches, candidate selection, and server launch
- `benchmark-minimal.py` — the same single-request, low-reasoning coding workload
  used by the Verda candidate protocol

To resume, create one B300 pod in `EUR-IS-1` with the pinned image, attach volume
`1u0a27qypv` at `/workspace`, expose port 8000, and provide `SGLANG_API_KEY` as a
secret environment variable. Start the known-safe profile first:

```bash
export DFLASH_PERF_EXPERIMENT=baseline
export DFLASH_DRAFT_TOKENS=8
exec /workspace/deploy/runpod-b300-dflash256-boot.sh
```

The launcher copies the verified ~197 GB checkpoint from network storage to
`/models-local` with 16 workers before loading it, then reuses
`/workspace/cache` for SGLang, Triton, TorchInductor, CUDA, and Hugging Face
caches. After `/health` is ready, benchmark the selected profile once:

```bash
python3 /workspace/deploy/benchmark-minimal.py \
  --output /workspace/dflash256-b300-${DFLASH_PERF_EXPERIMENT}-benchmark.json
```

Every launch re-hashes all eight runtime patches, compiles only the seven Python
patches as Python, verifies both checkpoint records, and writes a candidate-
specific server log. This makes a later B300 launch reproducible without keeping
an idle GPU pod alive.

The draft checkpoint is licensed `CC-BY-NC-ND-4.0`; review that license before commercial use.
