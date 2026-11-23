"""Deterministic, generated LFM2 MoE checkpoints for CPU qualification."""

from math import prod
import os
from pathlib import Path
import struct
import tempfile

from .gguf import ArrayInfo, GGUF, GGUFError, align_up, layout_tensors, pad_to, write_fixture_header


FIXTURE_MARKER = "rustyspork.fixture"
_MARKER_VALUE = "lfm2moe-q6k-v1"
_WIDTH = 256
_VOCAB_SIZE = 259
_TOP_K = 2
_F32 = 0
_Q6_K = 14


def _tensor_specs(experts: int) -> list[tuple[str, tuple[int, ...], int]]:
    width = _WIDTH
    specs = [
        ("token_embd.weight", (width, _VOCAB_SIZE), _F32),
        ("token_embd_norm.weight", (width,), _F32),
    ]
    for i in range(2):
        prefix = f"blk.{i}."
        specs.extend((prefix + name, shape, dtype) for name, shape, dtype in (
            ("attn_norm.weight", (width,), _F32),
            ("ffn_norm.weight", (width,), _F32),
            ("ffn_gate_inp.weight", (width, experts), _F32),
            ("ffn_gate_exps.weight", (width, width, experts), _Q6_K),
            ("ffn_up_exps.weight", (width, width, experts), _Q6_K),
            ("ffn_down_exps.weight", (width, width, experts), _Q6_K),
            ("exp_probs_b.bias", (experts,), _F32),
        ))
        if i == 0:
            specs.extend((prefix + name, shape, _F32) for name, shape in (
                ("shortconv.conv.weight", (3, width)),
                ("shortconv.in_proj.weight", (width, 3 * width)),
                ("shortconv.out_proj.weight", (width, width)),
            ))
        else:
            specs.extend((prefix + name, shape, _F32) for name, shape in (
                ("attn_q_norm.weight", (64,)),
                ("attn_k_norm.weight", (64,)),
                ("attn_q.weight", (width, width)),
                ("attn_k.weight", (width, width)),
                ("attn_v.weight", (width, width)),
                ("attn_output.weight", (width, width)),
            ))
    return specs


def _metadata(experts: int) -> dict:
    # GGUF metadata wire type IDs. Token kinds follow llama.cpp's SentencePiece
    # vocabulary; 256 byte tokens allow ordinary UTF-8 test prompts.
    tokens = ("<unk>", "<s>", "</s>") + tuple(f"<0x{i:02X}>" for i in range(256))
    return {
        "general.architecture": (8, "lfm2moe"),
        "general.name": (8, "RustySpork tiny LFM2 MoE"),
        "general.quantization_version": (4, 2),
        FIXTURE_MARKER: (8, _MARKER_VALUE),
        "lfm2moe.block_count": (4, 2),
        "lfm2moe.context_length": (4, 128),
        "lfm2moe.embedding_length": (4, _WIDTH),
        "lfm2moe.feed_forward_length": (4, _WIDTH),
        "lfm2moe.expert_count": (4, experts),
        "lfm2moe.expert_used_count": (4, _TOP_K),
        "lfm2moe.expert_feed_forward_length": (4, _WIDTH),
        "lfm2moe.expert_gating_func": (4, 2),
        "lfm2moe.leading_dense_block_count": (4, 0),
        "lfm2moe.attention.head_count": (4, 4),
        "lfm2moe.attention.head_count_kv": (9, (5, (0, 4))),
        "lfm2moe.attention.layer_norm_rms_epsilon": (6, 2 ** -16),
        "lfm2moe.shortconv.l_cache": (4, 3),
        "lfm2moe.vocab_size": (4, _VOCAB_SIZE),
        "tokenizer.ggml.model": (8, "llama"),
        "tokenizer.ggml.tokens": (9, (8, tokens)),
        "tokenizer.ggml.scores": (9, (6, (0.0,) * _VOCAB_SIZE)),
        "tokenizer.ggml.token_type": (9, (5, (2, 3, 3) + (6,) * 256)),
        "tokenizer.ggml.unknown_token_id": (4, 0),
        "tokenizer.ggml.bos_token_id": (4, 1),
        "tokenizer.ggml.eos_token_id": (4, 2),
        "tokenizer.ggml.add_bos_token": (7, True),
        "tokenizer.ggml.add_eos_token": (7, False),
    }


def check_fixture_layout(model: GGUF, experts: int) -> None:
    """Require the complete small-fixture metadata and tensor layout."""
    if not _TOP_K <= experts <= 8:
        raise GGUFError("Unsupported synthetic fixture expert count")
    expected_metadata = _metadata(experts)
    if model.metadata.keys() != expected_metadata.keys():
        raise GGUFError("Unexpected metadata keys in synthetic fixture")
    with model.open() as source:
        for name, (kind, expected) in expected_metadata.items():
            actual = model.metadata[name]
            record = model.metadata_records[name]
            if kind == 9:
                element_type, values = expected
                if (actual != ArrayInfo(element_type, len(values)) or record.nbytes > 8192
                        or model.metadata_array(name, max_elements=len(values), source_file=source) != values):
                    raise GGUFError(f"Unexpected synthetic fixture array {name}")
            elif record.type_id != kind or actual != expected:
                raise GGUFError(f"Unexpected synthetic fixture metadata {name}")
    actual_specs = {t.name: (t.shape, t.ggml_type) for t in model.tensors.values()}
    expected_specs = {name: (shape, dtype) for name, shape, dtype in _tensor_specs(experts)}
    if actual_specs != expected_specs:
        raise GGUFError("Unexpected tensors or tensor metadata in synthetic fixture")


def _f32_payload(name: str, shape: tuple[int, ...], layer: int) -> bytes | bytearray:
    if "norm.weight" in name:
        return struct.pack("<f", 1.0) * shape[0]
    if name.endswith("exp_probs_b.bias"):
        order = range(shape[0]) if layer == 0 else reversed(range(shape[0]))
        return b"".join(struct.pack("<f", i / 8) for i in order)
    if name.endswith("ffn_gate_inp.weight"):
        return b"".join(struct.pack("<f", (i + 1) / 4096) + bytes(4 * (shape[0] - 1))
                        for i in range(shape[1]))
    if name.endswith("shortconv.conv.weight"):
        return struct.pack("<3f", 0, 0, 1) * shape[1]
    payload = bytearray(4 * prod(shape))
    for row in range(shape[1]):
        struct.pack_into("<f", payload, 4 * (row * shape[0] + row % shape[0]), 0.125)
    return payload


def _q6k_payload(name: str, shape: tuple[int, ...], layer: int) -> bytearray:
    # Each 256-element block has ql = 1 in both nibbles, qh = 2 in all
    # two-bit lanes, and all subscales = 1. llama.cpp decodes q = 33 - 32,
    # hence all 256 values equal d. Vary d at row and expert boundaries.
    prefix = bytes([0x11]) * 128 + bytes([0xAA]) * 64 + bytes([1]) * 16
    role = {"ffn_gate_exps": 0, "ffn_up_exps": 1, "ffn_down_exps": 2}[name.split(".")[2]]
    output = bytearray()
    for expert in range(shape[2]):
        for row in range(shape[1]):
            scale = 1 + expert + 8 * role + 24 * layer + 48 * (row % 2)
            output.extend(prefix)
            output.extend(struct.pack("<e", scale / 16384))
    return output


def create_fixture(path: str | Path, expert_count: int = 4) -> Path:
    """Write the 4- or 8-expert fixture to a new path without overwriting files."""
    if type(expert_count) is not int or expert_count not in (4, 8):
        raise GGUFError("Fixture generation accepts 4 or 8 experts")
    path = Path(path).absolute()
    if path.exists():
        raise FileExistsError(path)
    tensors = layout_tensors(_tensor_specs(expert_count))
    staging = tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fixture-", suffix=".tmp", delete=False)
    try:
        with staging as out:
            data_offset = write_fixture_header(out, _metadata(expert_count), tensors)
            for tensor in tensors:
                layer = int(tensor.name.split(".")[1]) if tensor.name.startswith("blk.") else 0
                pad_to(out, data_offset + tensor.offset)
                if tensor.ggml_type == _Q6_K:
                    payload = _q6k_payload(tensor.name, tensor.shape, layer)
                else:
                    payload = _f32_payload(tensor.name, tensor.shape, layer)
                if len(payload) != tensor.nbytes:
                    raise GGUFError(f"Fixture length mismatch for {tensor.name}")
                out.write(payload)
            pad_to(out, align_up(out.tell(), 32))
            out.flush()
            os.fsync(out.fileno())
        # Publishing through an exclusive link cannot replace an existing path,
        # even if another process creates the destination during generation.
        os.link(staging.name, path)
    finally:
        Path(staging.name).unlink(missing_ok=True)
    return path
