#!/usr/bin/env python3
"""Run one useful streaming request and report TTFT plus decode throughput."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

PROMPT = """Implement a compact Python LRUCache class with get and put methods.
Use only the standard library, make both operations O(1), and return only the code."""

PERF_EXPERIMENTS = {
    "baseline",
    "replayssm-spec",
    "ngram-primary",
    "topk-v2",
    "kpool-metadata",
    "ingraph-metadata",
    "mhc-post-pre",
    "deepgemm-hc-prenorm",
    "cutedsl-paged-mqa",
    "blackwell-dsa",
    "b300-cyclepack",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not 1 <= args.max_tokens <= 256:
        raise ValueError("--max-tokens must be between 1 and 256")

    api_key = os.environ.get("SGLANG_API_KEY")
    if not api_key:
        raise RuntimeError("SGLANG_API_KEY must be set")

    perf_experiment = os.environ.get("DFLASH_PERF_EXPERIMENT", "baseline")
    if perf_experiment not in PERF_EXPERIMENTS:
        raise RuntimeError(f"unsupported DFLASH_PERF_EXPERIMENT: {perf_experiment}")
    output_path = args.output
    if output_path is None:
        output_path = f"/workspace/dflash256-{perf_experiment}-benchmark.json"

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
    ).encode()
    request = urllib.request.Request(
        args.endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    started = time.monotonic()
    first_token_at: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, object] | None = None

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            if (text or thought) and first_token_at is None:
                first_token_at = time.monotonic()
            content.append(text)
            reasoning.append(thought)

    finished = time.monotonic()
    if first_token_at is None:
        raise RuntimeError("stream returned no generated token")
    if usage is None:
        raise RuntimeError("stream returned no usage record")
    endpoint = urllib.parse.urlsplit(args.endpoint)
    server_info_url = urllib.parse.urlunsplit(
        (endpoint.scheme, endpoint.netloc, "/server_info", "", "")
    )
    server_info_request = urllib.request.Request(
        server_info_url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(
        server_info_request, timeout=args.timeout
    ) as server_info_response:
        server_info = json.load(server_info_response)
    internal_states = server_info.get("internal_states") or []
    avg_spec_accept_length = (
        internal_states[0].get("avg_spec_accept_length") if internal_states else None
    )
    if avg_spec_accept_length is not None:
        avg_spec_accept_length = round(float(avg_spec_accept_length), 4)

    draft_tokens = int(server_info["speculative_num_draft_tokens"])

    completion_tokens = int(usage["completion_tokens"])
    decode_seconds = max(finished - first_token_at, 1e-9)
    measured_decode_tokens = max(completion_tokens - 1, 0)
    result = {
        "verified": bool("".join(content) or "".join(reasoning)),
        "request_count": 1,
        "dflash_perf_experiment": perf_experiment,
        "prompt": "compact Python LRU cache",
        "max_completion_tokens": args.max_tokens,
        "ttft_ms": round((first_token_at - started) * 1000, 2),
        "decode_seconds": round(decode_seconds, 4),
        "decode_tokens_per_second": round(measured_decode_tokens / decode_seconds, 2),
        "dflash_draft_tokens": draft_tokens,
        "avg_spec_accept_length": avg_spec_accept_length,
        "visible_content": "".join(content),
        "usage": usage,
        "visible_content_chars": len("".join(content)),
        "reasoning_chars": len("".join(reasoning)),
    }
    if not result["verified"] or completion_tokens <= 0:
        raise RuntimeError(f"invalid generation result: {result}")

    encoded = json.dumps(result, indent=2) + "\n"
    if output_path:
        from pathlib import Path

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")
    print("MINIMAL_DFLASH256_BENCHMARK_VERIFIED")


if __name__ == "__main__":
    main()
