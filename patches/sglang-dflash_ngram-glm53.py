from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import triton
import triton.language as tl


def find_ngram_continuation_reference(
    history: Sequence[int],
    bonus_token: int,
    *,
    draft_token_num: int,
    gram_size: int = 8,
    window_size: int = 2048,
) -> Optional[list[int]]:
    """Return an observed continuation for the current suffix, or ``None``.

    This is the CPU specification for the Triton proposer below. The returned
    tokens are copied from prior context; the proposer never invents a cycle or
    bypasses target-model verification.
    """
    if draft_token_num < 2:
        return None
    if gram_size < 1 or window_size < gram_size:
        raise ValueError("invalid n-gram proposer configuration")

    context = [*history[-window_size:], int(bonus_token)]
    draft_count = draft_token_num - 1
    if len(context) < gram_size + draft_count:
        return None

    suffix = context[-gram_size:]
    latest_start = len(context) - gram_size - draft_count
    for start in range(latest_start, -1, -1):
        if context[start : start + gram_size] == suffix:
            return context[
                start + gram_size : start + gram_size + draft_count
            ]
    return None


@triton.jit
def _dflash_ngram_propose_kernel(
    history_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    bonus_tokens_ptr,
    draft_tokens_out_ptr,
    found_out_ptr,
    history_row_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    prefix_lens_stride: tl.constexpr,
    bonus_tokens_stride: tl.constexpr,
    draft_tokens_row_stride: tl.constexpr,
    found_stride: tl.constexpr,
    draft_token_num: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    GRAM_SIZE: tl.constexpr,
    SEARCH_BLOCK_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, SEARCH_BLOCK_SIZE)

    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride).to(
        tl.int64
    )
    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride).to(tl.int64)

    context_len = prefix_len + 1
    ring_start = tl.maximum(0, prefix_len - WINDOW_SIZE)
    candidate_start = ring_start + offsets
    draft_count = draft_token_num - 1
    latest_full_start = context_len - GRAM_SIZE - draft_count
    suffix_start = context_len - GRAM_SIZE

    match = (
        (offsets < WINDOW_SIZE)
        & (candidate_start <= latest_full_start)
        & (suffix_start >= ring_start)
    )
    history_row = history_ptr + req_idx * history_row_stride

    for gram_offset in range(GRAM_SIZE):
        candidate_pos = candidate_start + gram_offset
        candidate_slot = candidate_pos % WINDOW_SIZE
        candidate_token = tl.load(
            history_row + candidate_slot,
            mask=match & (candidate_pos < prefix_len),
            other=0,
        ).to(tl.int64)

        suffix_pos = suffix_start + gram_offset
        suffix_slot = suffix_pos % WINDOW_SIZE
        suffix_from_history = tl.load(
            history_row + suffix_slot,
            mask=(suffix_pos >= ring_start) & (suffix_pos < prefix_len),
            other=0,
        ).to(tl.int64)
        suffix_token = tl.where(
            suffix_pos == prefix_len, bonus_token, suffix_from_history
        )
        match = match & (candidate_token == suffix_token)

    latest_start = tl.max(tl.where(match, candidate_start, -1), axis=0)
    found = latest_start >= 0
    tl.store(found_out_ptr + row * found_stride, found.to(tl.int32))

    cols = tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = cols < draft_token_num
    continuation_pos = latest_start + GRAM_SIZE + cols - 1
    continuation_slot = continuation_pos % WINDOW_SIZE
    continuation_from_history = tl.load(
        history_row + continuation_slot,
        mask=found
        & (cols > 0)
        & output_mask
        & (continuation_pos >= ring_start)
        & (continuation_pos < prefix_len),
        other=0,
    ).to(tl.int64)
    continuation_token = tl.where(
        continuation_pos == prefix_len, bonus_token, continuation_from_history
    )
    output_token = tl.where(cols == 0, bonus_token, continuation_token)
    tl.store(
        draft_tokens_out_ptr + row * draft_tokens_row_stride + cols,
        output_token,
        mask=output_mask,
    )


@triton.jit
def _dflash_ngram_append_committed_kernel(
    history_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    committed_tokens_ptr,
    commit_lens_ptr,
    history_row_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    prefix_lens_stride: tl.constexpr,
    committed_tokens_row_stride: tl.constexpr,
    commit_lens_stride: tl.constexpr,
    draft_token_num: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride).to(
        tl.int64
    )
    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    commit_len = tl.load(commit_lens_ptr + row * commit_lens_stride).to(tl.int32)
    mask = (cols < draft_token_num) & (cols < commit_len)
    tokens = tl.load(
        committed_tokens_ptr + row * committed_tokens_row_stride + cols,
        mask=mask,
        other=0,
    )
    positions = prefix_len + cols
    slots = positions % WINDOW_SIZE
    tl.store(
        history_ptr + req_idx * history_row_stride + slots,
        tokens,
        mask=mask,
    )


def propose_dflash_ngram(
    *,
    history: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    draft_tokens_out: torch.Tensor,
    found_out: torch.Tensor,
    gram_size: int,
) -> bool:
    """Fill a DFlash chain from exact prior context and report an all-row hit."""
    batch_size, draft_token_num = draft_tokens_out.shape
    if batch_size == 0 or draft_token_num < 2:
        return False
    if history.ndim != 2 or history.dtype != torch.int64:
        raise ValueError("n-gram history must be a 2D int64 tensor")
    if not history.is_contiguous() or not draft_tokens_out.is_contiguous():
        raise ValueError("n-gram history and output must be contiguous")
    if req_pool_indices.ndim != 1 or prefix_lens.ndim != 1:
        raise ValueError("request indices and prefix lengths must be 1D")
    if bonus_tokens.numel() != batch_size or found_out.numel() < batch_size:
        raise ValueError("n-gram proposer batch buffers have inconsistent sizes")

    window_size = int(history.shape[1])
    if not 1 <= gram_size <= min(32, window_size):
        raise ValueError("n-gram size must be between 1 and min(32, window)")
    search_block_size = triton.next_power_of_2(window_size)
    output_block_size = triton.next_power_of_2(draft_token_num)
    _dflash_ngram_propose_kernel[(batch_size,)](
        history,
        req_pool_indices,
        prefix_lens,
        bonus_tokens.view(-1),
        draft_tokens_out,
        found_out,
        history.stride(0),
        req_pool_indices.stride(0),
        prefix_lens.stride(0),
        bonus_tokens.view(-1).stride(0),
        draft_tokens_out.stride(0),
        found_out.stride(0),
        draft_token_num=draft_token_num,
        WINDOW_SIZE=window_size,
        GRAM_SIZE=gram_size,
        SEARCH_BLOCK_SIZE=search_block_size,
        OUTPUT_BLOCK_SIZE=output_block_size,
        num_warps=8 if search_block_size >= 1024 else 4,
    )
    # This tiny sync decides whether the neural draft can be skipped. A partial
    # batch hit deliberately falls back for the whole batch; B300 serves bs=1.
    return bool(torch.all(found_out[:batch_size]).item())


def append_dflash_ngram_committed(
    *,
    history: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    committed_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Append only target-verified tokens to the per-request history ring."""
    batch_size, draft_token_num = committed_tokens.shape
    if batch_size == 0:
        return
    if history.ndim != 2 or history.dtype != torch.int64:
        raise ValueError("n-gram history must be a 2D int64 tensor")
    if not history.is_contiguous() or not committed_tokens.is_contiguous():
        raise ValueError("n-gram history and committed tokens must be contiguous")
    block_size = triton.next_power_of_2(draft_token_num)
    _dflash_ngram_append_committed_kernel[(batch_size,)](
        history,
        req_pool_indices,
        prefix_lens,
        committed_tokens,
        commit_lens,
        history.stride(0),
        req_pool_indices.stride(0),
        prefix_lens.stride(0),
        committed_tokens.stride(0),
        commit_lens.stride(0),
        draft_token_num=draft_token_num,
        WINDOW_SIZE=int(history.shape[1]),
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
