#!/usr/bin/env python3
"""Offline/static validation for the exact GLM-5.3-Flash SM120 deployment bundle.

This deliberately does not pretend to execute CUDA. It validates the real patched
source files, their lock hashes, the pinned checkpoint config, and the upstream
SGLang interfaces those patches consume.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
SGLANG_REV = "033446bb05f3"
HF_REV = "9e0d74e3cef17f634e84fb8e2223707e02616290"
DFLASH_REV = "bf582e4eacc1810f76656d1811693ff6c6737d2a"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "glm53-local-validator/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def validate_manifest_and_patches() -> None:
    manifest = json.loads(source("manifest.json"))
    check(manifest["hardware"]["gpu_count"] == 4, "bundle targets four GPUs")
    check(manifest["hardware"]["cuda_arch"] == "sm_120", "bundle targets SM120")
    check(manifest["server"]["moe_runner_backend"] == "flashinfer_cutlass", "SM120 CUTLASS MoE selected")
    check(manifest["server"]["dsa_decode_backend"] == "flashinfer_sparse_mla", "patched sparse-MLA decode selected")
    profile = json.loads(source("verda-dflash256-manifest.json"))
    check(profile["sglang"]["revision"].startswith(SGLANG_REV), "DFlash profile pins the image SGLang revision")
    check(profile["draft_model"]["revision"] == DFLASH_REV, "DFlash profile pins the draft checkpoint revision")
    check(profile["profile"]["context_length"] == 262144, "DFlash profile caps context at 256K")
    check(profile["profile"]["max_running_requests"] == 1, "DFlash profile limits concurrency to one request")
    check(profile["benchmark"]["default_request_count"] == 1, "DFlash benchmark contract permits one request")

    for patch in manifest["patches"]:
        data = (ROOT / patch["source"]).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        check(actual == patch["sha256"], f"locked hash matches {patch['source']}")
        compile(data, patch["source"], "exec")
        print(f"PASS: Python compiles {patch['source']}")


def validate_checkpoint_contract() -> None:
    url = (
        "https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/resolve/"
        f"{HF_REV}/config.json?download=1"
    )
    raw = fetch(url)
    manifest = json.loads(source("manifest.json"))
    check(hashlib.sha256(raw).hexdigest() == manifest["model"]["config_sha256"], "checkpoint config hash matches manifest")
    config = json.loads(raw)
    text = config["text_config"]
    check(config["architectures"] == ["Glm5NextForConditionalGeneration"], "checkpoint architecture is GLM5 Next conditional generation")
    check(text["qk_rope_head_dim"] == 0 and text["mla_use_nope"] is True, "checkpoint uses NoPE sparse MLA")
    check(text["index_topk"] == 2048 and text["index_kpool"] == 4, "DSA index dimensions are 2048 top-k / 4 k-pool")
    kda = text["linear_attn_config"]["kda_layers"]
    full = text["linear_attn_config"]["full_attn_layers"]
    check(len(kda) == 34 and len(full) == 11, "checkpoint has 34 KDA and 11 full-attention layers")
    check(sorted(kda + full) == list(range(text["num_hidden_layers"])), "attention layer partition covers all 45 layers")
    check(text["qk_nope_head_dim"] == text["v_head_dim"] == 256, "NoPE query/value head dimensions agree")
    profile = json.loads(source("verda-dflash256-manifest.json"))
    current_revision = profile["target_model"]["revision"]
    current_raw = fetch(
        "https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/resolve/"
        f"{current_revision}/config.json?download=1"
    )
    check(
        hashlib.sha256(current_raw).hexdigest()
        == profile["target_model"]["hashes"]["config.json"],
        "DFlash target config hash matches profile manifest",
    )
    ignored = json.loads(current_raw)["quantization_config"]["ignore"]
    check(
        "*.self_attn.fused_qkvbfg_a_proj" in ignored
        and "*.self_attn.fused_fg_b_proj" in ignored,
        "DFlash checkpoint keeps both fused KDA projections in BF16",
    )


def validate_upstream_interfaces() -> None:
    base = f"https://raw.githubusercontent.com/sgl-project/sglang/{SGLANG_REV}/python/sglang/srt"
    communicator = fetch(f"{base}/layers/communicator.py").decode()
    mhc = fetch(f"{base}/layers/communicator_mhc.py").decode()
    mhc_ops = fetch(f"{base}/../kernels/ops/layernorm/mhc.py").decode()
    glm_config = fetch(f"{base}/configs/glm5_next.py").decode()
    spec_info = fetch(f"{base}/speculative/spec_info.py").decode()
    dflash_hook = fetch(f"{base}/arg_groups/speculative_hook.py").decode()
    for name, text in (
        ("communicator.py", communicator),
        ("communicator_mhc.py", mhc),
        ("kernels/ops/layernorm/mhc.py", mhc_ops),
        ("configs/glm5_next.py", glm_config),
        ("speculative/spec_info.py", spec_info),
        ("arg_groups/speculative_hook.py", dflash_hook),
    ):
        ast.parse(text, filename=name)
        print(f"PASS: upstream interface source parses: {name}")

    check("def maybe_prefetch_next_full_attention_kv(" in communicator, "base communicator provides full-attention KV prefetch hook")
    check("class MHCLayerCommunicator(LayerCommunicator):" in mhc, "MHC communicator inherits the prefetch hook")
    check("def should_use_reduce_scatter(" in mhc, "MHC communicator provides reduce-scatter decision hook")
    check("def hc_contract(" in mhc_ops, "pinned SGLang exports the DFlash mHC contraction")
    check("def mhc_fused_post_pre(" in mhc_ops, "pinned SGLang exports fused mHC post/pre")
    check("def is_kda_layer(" in glm_config, "GLM config provides KDA layer classification")
    check("def full_attention_layer_ids(" in glm_config, "GLM config provides full-attention layer IDs")
    check("def mamba2_cache_params(" in glm_config, "GLM config provides hybrid-state cache parameters")
    check("if rope_scaling is None and rope_parameters is not None:" in glm_config, "GLM config safely aliases rope_parameters")
    check("DFLASH = auto()" in spec_info, "pinned SGLang provides generic DFlash")
    check("from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2" in spec_info, "pinned SGLang provides the DFlash V2 worker")
    check("def _handle_dflash(" in dflash_hook, "pinned SGLang provides DFlash argument resolution")

    draft_url = (
        "https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/resolve/"
        f"{DFLASH_REV}/config.json?download=1"
    )
    draft_raw = fetch(draft_url)
    profile = json.loads(source("verda-dflash256-manifest.json"))
    expected_hash = profile["draft_model"]["files"]["config.json"]["sha256"]
    check(hashlib.sha256(draft_raw).hexdigest() == expected_hash, "DFlash2 config hash matches profile manifest")
    draft = json.loads(draft_raw)
    check(draft["architectures"] == ["DFlash2DraftModel"], "draft checkpoint exports DFlash2")
    check(draft["dflash_config"]["block_size"] == 8, "draft checkpoint native block is eight tokens")
    check(draft["dflash_config"]["target_layer_ids"] == [5, 14, 24, 33, 42], "draft target-layer capture map is pinned")


def validate_patch_logic() -> None:
    model = source("patches/sglang-glm5_next-debug.py")
    dsa = source("patches/sglang-dsa_backend-glm53.py")
    flash = source("patches/sglang-flash_mla_sm120-glm53.py")
    quant = source("patches/sglang-modelopt-quant-sm120.py")
    nextn = source("patches/sglang-deepseek_nextn-glm53.py")
    communicator_mhc = source("patches/sglang-communicator_mhc-glm53.py")

    check("class Glm5NextForConditionalGeneration" in model, "patched model exports the checkpoint architecture")
    check("config.is_kda_layer(layer_id)" in model, "patched decoder dispatches KDA layers from config")
    check("maybe_prefetch_next_full_attention_kv(" in model, "patched model consumes the inherited prefetch hook")
    check("should_use_reduce_scatter(" in model, "patched model consumes the MHC reduce-scatter hook")
    check("from sglang.kernels.ops.layernorm.mhc import hc_contract" in model, "GLM patch imports the DFlash mHC contraction")
    check("def set_dflash_layers_to_capture(" in model, "GLM patch exposes the DFlash capture hook")
    check("if forward_batch.can_run_tbo and not self.dflash_capture:" in model, "DFlash capture keeps the eager layer loop")
    check("hidden_states if residual is None else hidden_states + residual" in model, "DFlash capture handles an absent residual")
    check("self.model.layers_to_capture = [val + 1 for val in layer_ids]" in model, "DFlash target layers map to completed layer outputs")
    check(
        "SGLANG_GLM53_FUSE_MHC_POST_PRE" in model
        and "_mhc_fused_post_pre_fn(" in model
        and "hc_fused_post_pre=(" in model,
        "GLM patch gates and wires fused mHC post/pre",
    )
    check(
        "hc_fused_post_pre: Optional[Callable]" in communicator_mhc
        and "self.hc_fused_post_pre(" in communicator_mhc,
        "MHC communicator dispatches the optional fused boundary",
    )
    check(
        "fused_projections_are_unquantized = quant_config is None or (" in model
        and 'f"{prefix}.fused_qkvbfg_a_proj"' in model
        and 'f"{prefix}.fused_fg_b_proj"' in model,
        "ModelOpt keeps explicitly excluded BF16 KDA projections fused",
    )
    check(
        "self.do_fuse_qkvbfg = (" in model and "quant_config=None" in model,
        "fused BF16 KDA projections bypass the global ModelOpt config",
    )
    check("skip_rope=config.qk_rope_head_dim == 0" in nextn, "NEXTN patch skips RoPE for NoPE checkpoint")
    check('"Glm5NextForConditionalGeneration"' in flash, "sparse-MLA patch recognizes GLM5 conditional architecture")
    check("if qk_rope_head_dim == 0:" in flash and "_flashinfer_nope_via_zero_rope" in flash, "sparse-MLA patch has explicit NoPE kernel bridge")
    check("q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)" in dsa, "DSA decode preserves already-concatenated NoPE query")
    check("if q_all is None:" in dsa and "if q_rope is not None:" in dsa, "DSA concatenates RoPE conditionally")
    check("flashinfer_cutlass" in quant.lower(), "ModelOpt quantization patch contains CUTLASS route")
    check("w13_input_scale" in quant and "w2_input_scale" in quant, "ModelOpt MoE registers both activation-scale tensors")

def validate_mhc_dispatch_contract() -> None:
    tree = ast.parse(
        source("patches/sglang-communicator_mhc-glm53.py"),
        filename="sglang-communicator_mhc-glm53.py",
    )
    state_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MHCState"
    )
    isolated = ast.Module(body=[state_node], type_ignores=[])
    ast.fix_missing_locations(isolated)

    class TorchStub:
        Tensor = object

        class nn:
            Module = object

    namespace = {
        "Callable": Callable,
        "Optional": Optional,
        "dataclass": dataclass,
        "torch": TorchStub,
    }
    exec(compile(isolated, "MHCState", "exec"), namespace)
    state_type = namespace["MHCState"]

    class FakeTensor:
        shape = (1,)

    calls = []
    post_result = FakeTensor()

    def hc_post(hidden_states, residual, h_res, h_post):
        calls.append(("post", hidden_states, residual, h_res, h_post))
        return post_result

    def hc_ffn_pre(hidden_states, norm_weight, norm_eps):
        calls.append(("pre", hidden_states, norm_weight, norm_eps))
        return FakeTensor(), "next_h_res", "next_h_post", False

    unfused = state_type(4, lambda *_: None, hc_ffn_pre, hc_post)
    unfused.h_res = "old_h_res"
    unfused.h_post = "old_h_post"
    _, residual = unfused.attn_to_mlp(FakeTensor(), "old_residual")
    check(
        [call[0] for call in calls] == ["post", "pre"]
        and residual is post_result
        and unfused.h_res == "next_h_res"
        and unfused.h_post == "next_h_post",
        "MHC baseline preserves separate post/pre dispatch",
    )

    def fail_unfused(*_args):
        raise AssertionError("unfused mHC callable reached from fused dispatch")

    def fused(hidden_states, residual, h_res, h_post, norm_weight, norm_eps):
        calls.append(
            ("fused", hidden_states, residual, h_res, h_post, norm_weight, norm_eps)
        )
        return FakeTensor(), "fused_residual", "fused_h_res", "fused_h_post", True

    fused_state = state_type(
        4,
        lambda *_: None,
        fail_unfused,
        fail_unfused,
        hc_fused_post_pre=fused,
    )
    fused_state.h_res = "old_h_res"
    fused_state.h_post = "old_h_post"
    _, residual = fused_state.attn_to_mlp(FakeTensor(), "old_residual")
    check(
        calls[-1][0] == "fused"
        and residual == "fused_residual"
        and fused_state.h_res == "fused_h_res"
        and fused_state.h_post == "fused_h_post",
        "MHC candidate uses only fused post/pre dispatch",
    )


def validate_glm_mhc_fusion_contract() -> None:
    tree = ast.parse(
        source("patches/sglang-glm5_next-debug.py"),
        filename="sglang-glm5_next-debug.py",
    )
    decoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextDecoderLayer"
    )
    method = next(
        node
        for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "hc_fused_post_pre"
    )
    harness = ast.ClassDef(
        name="FusionHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    isolated = ast.Module(body=[harness], type_ignores=[])
    ast.fix_missing_locations(isolated)

    class FakeTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

        def view(self, *shape):
            return FakeTensor(self.name, shape)

    captured = {}

    def fused_kernel(**kwargs):
        captured.update(kwargs)
        return (
            FakeTensor("residual", (2, 4, 4096)),
            FakeTensor("post", (2, 4, 1)),
            FakeTensor("comb", (2, 4, 4)),
            FakeTensor("layer", (2, 4096)),
        )

    namespace = {"_mhc_fused_post_pre_fn": fused_kernel}
    exec(compile(isolated, "Glm5NextDecoderLayer.hc_fused_post_pre", "exec"), namespace)
    layer = namespace["FusionHarness"]()

    class Config:
        hc_mult = 4
        rms_norm_eps = 1e-5
        hc_eps = 1e-6
        hc_sinkhorn_iters = 20

    layer.config = Config()
    layer.hc_ffn_fn = "ffn_fn"
    layer.hc_ffn_scale = "ffn_scale"
    layer.hc_ffn_base = "ffn_base"
    norm_weight = object()
    result = layer.hc_fused_post_pre(
        FakeTensor("hidden", (2, 4096)),
        FakeTensor("old_residual", (2, 16384)),
        FakeTensor("old_h_res", (2, 16)),
        FakeTensor("old_h_post", (2, 4)),
        norm_weight,
        1e-5,
    )
    check(
        [tensor.name for tensor in result[:4]]
        == ["layer", "residual", "comb", "post"]
        and result[4] is True
        and captured["fn"] == "ffn_fn"
        and captured["norm_weight"] is norm_weight
        and captured["residual"].shape == (2, 4, 4096)
        and captured["comb_res_mix"].shape == (2, 4, 4),
        "GLM fused wrapper preserves communicator state ordering and shapes",
    )


def validate_selector_contract() -> None:
    selector_path = ROOT / "select-dflash-experiment.py"
    spec = importlib.util.spec_from_file_location(
        "select_dflash_experiment", selector_path
    )
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)

    with tempfile.TemporaryDirectory() as directory:
        env_file = Path(directory) / "sglang.env"
        env_file.write_text(
            "SGLANG_API_KEY=test-value\n"
            "DFLASH_PERF_EXPERIMENT=baseline\n"
            "DFLASH_PERF_EXPERIMENT=topk-v2\n",
            encoding="utf-8",
        )
        env_file.chmod(0o640)
        selector.replace_experiment(env_file, "mhc-post-pre")
        lines = env_file.read_text(encoding="utf-8").splitlines()
        check(
            lines == [
                "SGLANG_API_KEY=test-value",
                "DFLASH_PERF_EXPERIMENT=mhc-post-pre",
            ],
            "experiment selector preserves secrets and replaces duplicate profile keys",
        )
        check(
            env_file.stat().st_mode & 0o777 == 0o640,
            "experiment selector preserves env-file permissions",
        )


def docker_cmd_tokens() -> list[str]:
    lines = source("Dockerfile").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("CMD "))
    logical = []
    for line in lines[start:]:
        logical.append(line[:-1].rstrip() if line.rstrip().endswith("\\") else line)
        if not line.rstrip().endswith("\\"):
            break
    command = " ".join(logical)[4:].strip()
    tokens = json.loads(command)
    check(all(isinstance(token, str) for token in tokens), "Docker CMD is valid JSON exec form")
    return tokens


def validate_launch_contract() -> None:
    tokens = docker_cmd_tokens()
    dockerfile = source("Dockerfile")
    check(
        "COPY patches/sglang-communicator_mhc-glm53.py "
        "/sgl-workspace/sglang/python/sglang/srt/layers/communicator_mhc.py"
        in dockerfile,
        "Docker image installs the mHC communicator patch",
    )
    required_pairs = {
        "--tp-size": "4",
        "--ep-size": "4",
        "--quantization": "modelopt_fp4",
        "--attention-backend": "dsa",
        "--dsa-prefill-backend": "flashinfer_sparse_mla",
        "--dsa-decode-backend": "flashinfer_sparse_mla",
        "--linear-attn-backend": "triton",
        "--kv-cache-dtype": "fp8_e4m3",
        "--moe-runner-backend": "flashinfer_cutlass",
        "--reasoning-parser": "glm45",
        "--tool-call-parser": "glm47",
    }
    for flag, expected in required_pairs.items():
        position = tokens.index(flag)
        check(tokens[position + 1] == expected, f"Docker launch sets {flag}={expected}")
    for flag in ("--disable-shared-experts-fusion", "--enable-multimodal"):
        check(flag in tokens, f"Docker launch enables required flag {flag}")

def validate_dflash_profile() -> None:
    profile = json.loads(source("verda-dflash256-manifest.json"))
    boot = source("verda-dflash256-boot.sh")
    required_pairs = {
        "--tp-size": "4",
        "--ep-size": "4",
        "--context-length": "262144",
        "--max-total-tokens": "262144",
        "--max-running-requests": "1",
        "--chunked-prefill-size": "4096",
        "--max-prefill-tokens": "4096",
        "--mem-fraction-static": "0.85",
        "--mamba-full-memory-ratio": "2",
        "--cuda-graph-max-bs-decode": "1",
        "--speculative-algorithm": "DFLASH",
        "--speculative-draft-window-size": "2048",
    }
    for flag, expected in required_pairs.items():
        check(f"{flag} {expected}" in boot, f"DFlash launch sets {flag}={expected}")
    check(
        "DFLASH_DRAFT_TOKENS=${DFLASH_DRAFT_TOKENS:-8}" in boot
        and '8|16|32|64)' in boot
        and '--speculative-num-draft-tokens "$DFLASH_DRAFT_TOKENS"' in boot,
        "DFlash launch defaults to width eight and admits only measured candidates",
    )
    check('--speculative-draft-model-path "$DRAFT"' in boot, "DFlash launch uses the verified local draft")
    check("--speculative-adaptive" not in boot, "DFlash launch does not inherit NEXTN adaptive settings")
    check("exec python3 -m sglang.launch_server" in boot, "DFlash launcher hands signals to SGLang")

    experiment_contract = profile["performance_experiments"]
    experiment_names = {
        "baseline",
        "replayssm-spec",
        "topk-v2",
        "kpool-metadata",
        "ingraph-metadata",
        "mhc-post-pre",
    }
    check(
        experiment_contract["default"] == "baseline"
        and experiment_contract["one_candidate_per_restart"] is True
        and set(experiment_contract["candidates"]) == experiment_names - {"baseline"},
        "DFlash manifest locks isolated performance candidates",
    )
    check(
        "DFLASH_PERF_EXPERIMENT=${DFLASH_PERF_EXPERIMENT:-baseline}" in boot
        and all(f"  {name})" in boot for name in experiment_names),
        "DFlash launcher validates every performance candidate",
    )
    check(
        "EXPERIMENT_ARGS+=(--enable-linear-replayssm-spec)" in boot
        and "export SGLANG_OPT_USE_TOPK_V2=1" in boot
        and "export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1" in boot
        and "export SGLANG_EXPERIMENTAL_DSA_INGRAPH_VERIFY_METADATA=1" in boot
        and "export SGLANG_GLM53_FUSE_MHC_POST_PRE=1" in boot,
        "DFlash launcher maps every candidate to its kernel control",
    )
    check(
        '"/workspace/dflash256-${DFLASH_PERF_EXPERIMENT}.log"' in boot,
        "DFlash launcher separates candidate logs",
    )

    stager = source("stage-dflash2.py")
    ast.parse(stager, filename="stage-dflash2.py")
    check(profile["draft_model"]["revision"] in stager, "stager pins the manifest draft revision")
    for expected in profile["draft_model"]["files"].values():
        check(expected["sha256"] in stager, "stager pins a manifest draft-file hash")

    unit = source("sglang-glm-dflash256.service")
    check(profile["base_image"] in unit, "systemd profile pins the exact runtime image")
    check("--network=host" in unit and "--entrypoint /bin/bash" in unit, "systemd profile preserves loopback serving and shell entrypoint")

    benchmark = source("benchmark-minimal.py")
    ast.parse(benchmark, filename="benchmark-minimal.py")
    check(benchmark.count("urllib.request.urlopen(request") == 1, "minimal benchmark makes exactly one generation request")
    check('\"request_count\": 1' in benchmark, "minimal benchmark reports its single-request contract")
    check(
        '"/server_info"' in benchmark
        and '"avg_spec_accept_length"' in benchmark
        and '"dflash_draft_tokens"' in benchmark
        and '"dflash_perf_experiment"' in benchmark,
        "minimal benchmark records candidate, width, and acceptance",
    )
    check(
        'f"/workspace/dflash256-{perf_experiment}-benchmark.json"' in benchmark,
        "minimal benchmark retains one result per candidate",
    )
    check('default=256' in benchmark, "minimal benchmark caps completion at 256 tokens")
    check("attractor" not in benchmark.lower(), "minimal benchmark contains no attractor workload")

    selector = source("select-dflash-experiment.py")
    ast.parse(selector, filename="select-dflash-experiment.py")
    check(
        all(f'"{name}"' in selector for name in experiment_names)
        and "os.replace(temporary_name, path)" in selector,
        "experiment selector admits the manifest candidates and updates atomically",
    )

    wrapper = source("stage-dflash2.sh")
    check("refusing an implicit large pull" in wrapper, "CPU stager refuses a surprise image pull")
    check(profile["base_image"] in wrapper, "CPU stager pins the exact cached image")


if __name__ == "__main__":
    try:
        validate_manifest_and_patches()
        validate_checkpoint_contract()
        validate_upstream_interfaces()
        validate_patch_logic()
        validate_mhc_dispatch_contract()
        validate_glm_mhc_fusion_contract()
        validate_selector_contract()
        validate_launch_contract()
        validate_dflash_profile()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise
    print("ALL LOCAL STATIC/INTERFACE CHECKS PASSED")
    print("Static checks do not replace the recorded live SM120 verification.")
