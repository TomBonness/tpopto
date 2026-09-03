from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import triton
import triton.language as tl


_MAX_GRAM_SIZE = 16
_MAX_HASH_PROBES = 64
_INDEX_BLOCK_SIZE = 128


def find_ngram_continuation_reference(
    history: Sequence[int],
    bonus_token: int,
    *,
    draft_token_num: int,
    gram_sizes: Sequence[int] = (8, 6, 4),
    window_size: int = 8192,
) -> Optional[list[int]]:
    """Return the highest-priority complete continuation observed in context.

    This CPU specification mirrors the GPU hash-index lookup. Each order searches
    newest-to-oldest, and a shorter order is attempted only when every longer
    order misses. Returned tokens are copied from prior context; target-model
    verification remains authoritative.
    """
    if draft_token_num < 2:
        return None
    orders = tuple(int(order) for order in gram_sizes)
    if (
        not orders
        or len(set(orders)) != len(orders)
        or any(order < 1 or order > min(_MAX_GRAM_SIZE, window_size) for order in orders)
    ):
        raise ValueError("invalid prompt-lookup gram sizes")

    context = [*history[-window_size:], int(bonus_token)]
    draft_count = draft_token_num - 1
    for gram_size in orders:
        if len(context) < gram_size + draft_count:
            continue
        suffix = context[-gram_size:]
        latest_start = len(context) - gram_size - draft_count
        for start in range(latest_start, -1, -1):
            if context[start : start + gram_size] == suffix:
                return context[
                    start + gram_size : start + gram_size + draft_count
                ]
    return None


@triton.jit
def _hash_history_ngram(
    history_row,
    start,
    gram_size,
    valid,
    WINDOW_SIZE: tl.constexpr,
    MAX_GRAM_SIZE: tl.constexpr,
):
    value = start * 0 + 1469598103934665603
    for offset in range(MAX_GRAM_SIZE):
        active = valid & (offset < gram_size)
        position = start + offset
        token = tl.load(
            history_row + position % WINDOW_SIZE,
            mask=active,
            other=0,
        ).to(tl.int64)
        mixed = (value ^ (token + 0x9E3779B9)) * 1099511628211
        value = tl.where(offset < gram_size, mixed, value)
    value = value ^ (gram_size.to(tl.int64) * 0x517CC1B727220A95)
    value = value & 0x7FFFFFFFFFFFFFFF
    return tl.where(value == 0, 1, value)


@triton.jit
def _insert_hash_index(
    index_keys_ptr,
    index_values_ptr,
    key,
    start,
    valid,
    INDEX_CAPACITY: tl.constexpr,
    MAX_HASH_PROBES: tl.constexpr,
):
    """Insert with linear probing; INDEX_CAPACITY is an overflow sentinel slot."""
    active = valid
    base_slot = key & (INDEX_CAPACITY - 1)
    for probe in range(MAX_HASH_PROBES):
        slot = (base_slot + probe) & (INDEX_CAPACITY - 1)
        safe_slot = tl.where(active, slot, INDEX_CAPACITY)
        safe_key = tl.where(active, key, 0)
        previous = tl.atomic_cas(
            index_keys_ptr + safe_slot,
            0,
            safe_key,
            sem="relaxed",
            scope="gpu",
        )
        claimed = active & ((previous == 0) | (previous == key))
        value_slot = tl.where(claimed, slot, INDEX_CAPACITY)
        value = tl.where(claimed, start, -1)
        tl.atomic_max(
            index_values_ptr + value_slot,
            value,
            sem="relaxed",
            scope="gpu",
        )
        active = active & ~claimed


@triton.jit
def _build_dflash_prompt_index_kernel(
    history_ptr,
    gram_sizes_ptr,
    index_keys_ptr,
    index_values_ptr,
    req_idx,
    prefix_len,
    history_row_stride: tl.constexpr,
    NUM_ORDERS: tl.constexpr,
    draft_token_num: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    MAX_GRAM_SIZE: tl.constexpr,
    MAX_HASH_PROBES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    order_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gram_size = tl.load(gram_sizes_ptr + order_idx).to(tl.int32)

    prefix_len = prefix_len.to(tl.int64)
    ring_start = tl.maximum(0, prefix_len - WINDOW_SIZE)
    start = ring_start + offsets
    draft_count = draft_token_num - 1
    latest_mature_start = prefix_len - gram_size - draft_count
    valid = (
        (order_idx < NUM_ORDERS)
        & (offsets < WINDOW_SIZE)
        & (start <= latest_mature_start)
    )
    history_row = history_ptr + req_idx.to(tl.int64) * history_row_stride
    key = _hash_history_ngram(
        history_row,
        start,
        gram_size,
        valid,
        WINDOW_SIZE=WINDOW_SIZE,
        MAX_GRAM_SIZE=MAX_GRAM_SIZE,
    )
    _insert_hash_index(
        index_keys_ptr,
        index_values_ptr,
        key,
        start,
        valid,
        INDEX_CAPACITY=INDEX_CAPACITY,
        MAX_HASH_PROBES=MAX_HASH_PROBES,
    )


@triton.jit
def _dflash_prompt_lookup_kernel(
    history_ptr,
    gram_sizes_ptr,
    index_keys_ptr,
    index_values_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    bonus_tokens_ptr,
    candidate_chains_ptr,
    order_found_ptr,
    history_row_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    prefix_lens_stride: tl.constexpr,
    bonus_tokens_stride: tl.constexpr,
    candidate_order_stride: tl.constexpr,
    candidate_row_stride: tl.constexpr,
    found_order_stride: tl.constexpr,
    found_row_stride: tl.constexpr,
    NUM_ORDERS: tl.constexpr,
    draft_token_num: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    MAX_GRAM_SIZE: tl.constexpr,
    MAX_HASH_PROBES: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    order_idx = tl.program_id(0)
    row = tl.program_id(1)
    gram_size = tl.load(gram_sizes_ptr + order_idx).to(tl.int32)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride).to(
        tl.int64
    )
    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride).to(tl.int64)
    history_row = history_ptr + req_idx * history_row_stride

    context_len = prefix_len + 1
    ring_start = tl.maximum(0, prefix_len - WINDOW_SIZE)
    suffix_start = context_len - gram_size
    suffix_valid = suffix_start >= ring_start

    suffix_hash = suffix_start * 0 + 1469598103934665603
    for offset in range(MAX_GRAM_SIZE):
        active = suffix_valid & (offset < gram_size)
        suffix_pos = suffix_start + offset
        history_token = tl.load(
            history_row + suffix_pos % WINDOW_SIZE,
            mask=active & (suffix_pos < prefix_len),
            other=0,
        ).to(tl.int64)
        suffix_token = tl.where(suffix_pos == prefix_len, bonus_token, history_token)
        mixed = (suffix_hash ^ (suffix_token + 0x9E3779B9)) * 1099511628211
        suffix_hash = tl.where(offset < gram_size, mixed, suffix_hash)
    suffix_hash = suffix_hash ^ (
        gram_size.to(tl.int64) * 0x517CC1B727220A95
    )
    suffix_hash = suffix_hash & 0x7FFFFFFFFFFFFFFF
    suffix_hash = tl.where(suffix_hash == 0, 1, suffix_hash)

    base_slot = suffix_hash & (INDEX_CAPACITY - 1)
    indexed_start = tl.full((), -1, tl.int64)
    searching = suffix_valid
    for probe in range(MAX_HASH_PROBES):
        slot = (base_slot + probe) & (INDEX_CAPACITY - 1)
        key = tl.load(index_keys_ptr + slot)
        key_match = searching & (key == suffix_hash)
        value = tl.load(index_values_ptr + slot)
        indexed_start = tl.where(key_match, value, indexed_start)
        searching = searching & (key != 0) & ~key_match

    draft_count = draft_token_num - 1
    indexed_match = (
        (indexed_start >= ring_start)
        & (indexed_start + gram_size + draft_count <= context_len)
    )
    direct_start = context_len - gram_size - draft_count
    direct_match = (direct_start >= ring_start) & (direct_start >= 0)
    for offset in range(MAX_GRAM_SIZE):
        compare = offset < gram_size
        suffix_pos = suffix_start + offset
        suffix_from_history = tl.load(
            history_row + suffix_pos % WINDOW_SIZE,
            mask=compare & (suffix_pos < prefix_len),
            other=0,
        ).to(tl.int64)
        suffix_token = tl.where(
            suffix_pos == prefix_len, bonus_token, suffix_from_history
        )
        indexed_token = tl.load(
            history_row + (indexed_start + offset) % WINDOW_SIZE,
            mask=compare & indexed_match,
            other=0,
        ).to(tl.int64)
        direct_token = tl.load(
            history_row + (direct_start + offset) % WINDOW_SIZE,
            mask=compare & direct_match,
            other=0,
        ).to(tl.int64)
        indexed_match = indexed_match & (~compare | (indexed_token == suffix_token))
        direct_match = direct_match & (~compare | (direct_token == suffix_token))

    selected_start = tl.where(direct_match, direct_start, indexed_start)
    found = direct_match | indexed_match
    tl.store(
        order_found_ptr
        + order_idx * found_order_stride
        + row * found_row_stride,
        found.to(tl.int32),
    )

    cols = tl.arange(0, OUTPUT_BLOCK_SIZE)
    output_mask = cols < draft_token_num
    continuation_pos = selected_start + gram_size + cols - 1
    continuation_from_history = tl.load(
        history_row + continuation_pos % WINDOW_SIZE,
        mask=found
        & (cols > 0)
        & output_mask
        & (continuation_pos < prefix_len),
        other=0,
    ).to(tl.int64)
    continuation_token = tl.where(
        continuation_pos == prefix_len, bonus_token, continuation_from_history
    )
    output_token = tl.where(cols == 0, bonus_token, continuation_token)
    tl.store(
        candidate_chains_ptr
        + order_idx * candidate_order_stride
        + row * candidate_row_stride
        + cols,
        output_token,
        mask=output_mask,
    )


@triton.jit
def _select_prompt_lookup_chain_kernel(
    candidate_chains_ptr,
    order_found_ptr,
    draft_tokens_out_ptr,
    selected_found_ptr,
    candidate_order_stride: tl.constexpr,
    candidate_row_stride: tl.constexpr,
    found_order_stride: tl.constexpr,
    found_row_stride: tl.constexpr,
    draft_tokens_row_stride: tl.constexpr,
    selected_found_stride: tl.constexpr,
    NUM_ORDERS: tl.constexpr,
    draft_token_num: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    selected_order = tl.full((), -1, tl.int32)
    for order_idx in range(NUM_ORDERS):
        found = tl.load(
            order_found_ptr
            + order_idx * found_order_stride
            + row * found_row_stride
        )
        selected_order = tl.where(
            (selected_order < 0) & (found != 0), order_idx, selected_order
        )
    selected = selected_order >= 0
    tl.store(
        selected_found_ptr + row * selected_found_stride,
        selected.to(tl.int32),
    )

    cols = tl.arange(0, OUTPUT_BLOCK_SIZE)
    mask = selected & (cols < draft_token_num)
    tokens = tl.load(
        candidate_chains_ptr
        + selected_order.to(tl.int64) * candidate_order_stride
        + row * candidate_row_stride
        + cols,
        mask=mask,
        other=0,
    )
    tl.store(
        draft_tokens_out_ptr + row * draft_tokens_row_stride + cols,
        tokens,
        mask=mask,
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
    tl.store(
        history_ptr + req_idx * history_row_stride + positions % WINDOW_SIZE,
        tokens,
        mask=mask,
    )


@triton.jit
def _update_dflash_prompt_index_kernel(
    history_ptr,
    gram_sizes_ptr,
    index_keys_ptr,
    index_values_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    commit_lens_ptr,
    history_row_stride: tl.constexpr,
    req_pool_indices_stride: tl.constexpr,
    prefix_lens_stride: tl.constexpr,
    commit_lens_stride: tl.constexpr,
    NUM_ORDERS: tl.constexpr,
    draft_token_num: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    INDEX_CAPACITY: tl.constexpr,
    MAX_GRAM_SIZE: tl.constexpr,
    MAX_HASH_PROBES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    order_idx = tl.program_id(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK_SIZE)
    gram_size = tl.load(gram_sizes_ptr + order_idx).to(tl.int32)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride).to(
        tl.int64
    )
    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride).to(tl.int64)
    commit_len = tl.load(commit_lens_ptr + row * commit_lens_stride).to(tl.int32)
    draft_count = draft_token_num - 1

    mature_end = prefix_len + cols
    start = mature_end + 1 - gram_size - draft_count
    new_prefix_len = prefix_len + commit_len
    ring_start = tl.maximum(0, new_prefix_len - WINDOW_SIZE)
    valid = (
        (cols < draft_token_num)
        & (cols < commit_len)
        & (start >= ring_start)
    )
    history_row = history_ptr + req_idx * history_row_stride
    key = _hash_history_ngram(
        history_row,
        start,
        gram_size,
        valid,
        WINDOW_SIZE=WINDOW_SIZE,
        MAX_GRAM_SIZE=MAX_GRAM_SIZE,
    )
    _insert_hash_index(
        index_keys_ptr,
        index_values_ptr,
        key,
        start,
        valid,
        INDEX_CAPACITY=INDEX_CAPACITY,
        MAX_HASH_PROBES=MAX_HASH_PROBES,
    )


def build_dflash_prompt_index(
    *,
    history: torch.Tensor,
    req_pool_idx: int,
    prefix_len: int,
    gram_sizes: torch.Tensor,
    index_keys: torch.Tensor,
    index_values: torch.Tensor,
    draft_token_num: int,
) -> None:
    """Rebuild the bounded prompt index after prefill for one active request."""
    if prefix_len <= 0:
        return
    num_orders = int(gram_sizes.numel())
    window_size = int(history.shape[1])
    index_capacity = int(index_keys.numel()) - 1
    grid = (num_orders, triton.cdiv(window_size, _INDEX_BLOCK_SIZE))
    _build_dflash_prompt_index_kernel[grid](
        history,
        gram_sizes,
        index_keys,
        index_values,
        req_pool_idx,
        prefix_len,
        history.stride(0),
        NUM_ORDERS=num_orders,
        draft_token_num=draft_token_num,
        WINDOW_SIZE=window_size,
        INDEX_CAPACITY=index_capacity,
        MAX_GRAM_SIZE=_MAX_GRAM_SIZE,
        MAX_HASH_PROBES=_MAX_HASH_PROBES,
        BLOCK_SIZE=_INDEX_BLOCK_SIZE,
        num_warps=4,
    )


def propose_dflash_ngram(
    *,
    history: torch.Tensor,
    gram_sizes: torch.Tensor,
    index_keys: torch.Tensor,
    index_values: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    candidate_chains: torch.Tensor,
    order_found: torch.Tensor,
    draft_tokens_out: torch.Tensor,
    selected_found: torch.Tensor,
) -> bool:
    """Select the longest configured exact prompt continuation for each row."""
    batch_size, draft_token_num = draft_tokens_out.shape
    num_orders = int(gram_sizes.numel())
    if batch_size == 0 or draft_token_num < 2:
        return False
    if history.ndim != 2 or history.dtype != torch.int64:
        raise ValueError("prompt history must be a 2D int64 tensor")
    if not history.is_contiguous() or not draft_tokens_out.is_contiguous():
        raise ValueError("prompt history and output must be contiguous")
    if candidate_chains.shape[:2] != (num_orders, batch_size):
        raise ValueError("prompt candidate buffer has an incompatible shape")
    if order_found.shape[:2] != (num_orders, batch_size):
        raise ValueError("prompt found buffer has an incompatible shape")
    index_capacity = int(index_keys.numel()) - 1
    if index_capacity < 1 or index_capacity & (index_capacity - 1):
        raise ValueError("prompt hash-index capacity must be a power of two")

    output_block_size = triton.next_power_of_2(draft_token_num)
    _dflash_prompt_lookup_kernel[(num_orders, batch_size)](
        history,
        gram_sizes,
        index_keys,
        index_values,
        req_pool_indices,
        prefix_lens,
        bonus_tokens.view(-1),
        candidate_chains,
        order_found,
        history.stride(0),
        req_pool_indices.stride(0),
        prefix_lens.stride(0),
        bonus_tokens.view(-1).stride(0),
        candidate_chains.stride(0),
        candidate_chains.stride(1),
        order_found.stride(0),
        order_found.stride(1),
        NUM_ORDERS=num_orders,
        draft_token_num=draft_token_num,
        WINDOW_SIZE=int(history.shape[1]),
        INDEX_CAPACITY=index_capacity,
        MAX_GRAM_SIZE=_MAX_GRAM_SIZE,
        MAX_HASH_PROBES=_MAX_HASH_PROBES,
        OUTPUT_BLOCK_SIZE=output_block_size,
        num_warps=1,
    )
    _select_prompt_lookup_chain_kernel[(batch_size,)](
        candidate_chains,
        order_found,
        draft_tokens_out,
        selected_found,
        candidate_chains.stride(0),
        candidate_chains.stride(1),
        order_found.stride(0),
        order_found.stride(1),
        draft_tokens_out.stride(0),
        selected_found.stride(0),
        NUM_ORDERS=num_orders,
        draft_token_num=draft_token_num,
        OUTPUT_BLOCK_SIZE=output_block_size,
        num_warps=1,
    )
    # One tiny sync decides whether the neural draft can be skipped. B300 serves
    # one request, and a partial batch hit intentionally falls back as a unit.
    return bool(torch.all(selected_found[:batch_size]).item())


def append_dflash_ngram_committed(
    *,
    history: torch.Tensor,
    gram_sizes: torch.Tensor,
    index_keys: torch.Tensor,
    index_values: torch.Tensor,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    committed_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Append verified input tokens, then index newly mature continuations."""
    batch_size, draft_token_num = committed_tokens.shape
    if batch_size == 0:
        return
    if history.ndim != 2 or history.dtype != torch.int64:
        raise ValueError("prompt history must be a 2D int64 tensor")
    if not history.is_contiguous() or not committed_tokens.is_contiguous():
        raise ValueError("prompt history and committed tokens must be contiguous")
    num_orders = int(gram_sizes.numel())
    index_capacity = int(index_keys.numel()) - 1
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
    _update_dflash_prompt_index_kernel[(num_orders, batch_size)](
        history,
        gram_sizes,
        index_keys,
        index_values,
        req_pool_indices,
        prefix_lens,
        commit_lens,
        history.stride(0),
        req_pool_indices.stride(0),
        prefix_lens.stride(0),
        commit_lens.stride(0),
        NUM_ORDERS=num_orders,
        draft_token_num=draft_token_num,
        WINDOW_SIZE=int(history.shape[1]),
        INDEX_CAPACITY=index_capacity,
        MAX_GRAM_SIZE=_MAX_GRAM_SIZE,
        MAX_HASH_PROBES=_MAX_HASH_PROBES,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
