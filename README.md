# Exact GLM-5.3-Flash-NVFP4 SGLang bundle for 4x RTX PRO 6000 Blackwell (SM120)

Locked, independently buildable representation of the live Pop!_OS deployment (`glm-5.3-flash-sglang-sm120`). Uses the exact `lmsysorg/sglang:glm-5.3-flash` image digest and preserves the effective server arguments and environment from the validated running container.

The 182 GiB checkpoint is intentionally **not** copied into the image. Compose mounts the verified local checkpoint read-only from `/mnt/llm_models/GLM-5.3-Flash-NVFP4` and reuses the writable SGLang cache at `/home/ser/.cache/sglang-glm53`. The six SM120 compatibility patches are **baked into the image** at their editable-install import paths, so no bind-mounted code is needed.

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

Prepared files:

- `verda-dflash256-manifest.json` — revisions, hashes, retained Verda resource IDs, and launch contract
- `stage-dflash2.py` — resumable pinned download with full SHA-256 verification
- `stage-dflash2.sh` — runs the stager using the already-cached SGLang image and refuses an implicit large image pull
- `verda-dflash256-boot.sh` — validates both checkpoints and every runtime patch before starting SGLang
- `sglang-glm-dflash256.service` — replacement systemd unit
- `benchmark-minimal.py` — exactly one useful coding request, capped at 256 completion tokens, plus a non-generating `/server_info` read for active width and optional average acceptance

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

The draft checkpoint is licensed `CC-BY-NC-ND-4.0`; review that license before commercial use.
