"""Architectural contracts for the expert axes RustySpork can identify."""

from dataclasses import dataclass
import re

from .gguf import GGUF, GGUFError, UINT32


@dataclass(frozen=True)
class ExpertTensor:
    name: str
    axis: int
    role: str


@dataclass(frozen=True)
class MoeLayer:
    index: int
    tensors: tuple[ExpertTensor, ...]


@dataclass(frozen=True)
class MoeLayout:
    architecture: str
    expert_count_key: str
    expert_count: int
    top_k: int
    layers: tuple[MoeLayer, ...]
    can_compact: bool
    blocker: str | None


# These prefixes cover routed-expert values, scalars and biases, including
# alternative layouts that must fail validation. Shared experts are independent.
_ROUTED_NAME = re.compile(
    r"^blk\.\d+\.(?:ffn_[a-z_]*(?:_exp|_exps)(?=[._])|ffn_gate_inp\.|exp_probs_b[._]|expert_)"
)
_F32 = 0


def _integer(model: GGUF, key: str, minimum: int = 1) -> int:
    value = model.metadata.get(key)
    if type(value) is not int or value < minimum or model.metadata_records[key].type_id != UINT32:
        raise GGUFError(f"Expected UINT32 metadata {key} >= {minimum}")
    return value


def _check_tensor(model: GGUF, name: str, shape: tuple[int, ...], kind: int | None = None) -> None:
    tensor = model.tensors.get(name)
    if tensor is None:
        raise GGUFError(f"Missing required architecture tensor {name}")
    if tensor.shape != shape:
        raise GGUFError(f"{name} has shape {tensor.shape}, expected GGML shape {shape}")
    if kind is not None and tensor.ggml_type != kind:
        raise GGUFError(f"{name} has GGML type {tensor.ggml_type}, expected {kind}")


def inspect_moe(model: GGUF) -> MoeLayout | None:
    """Validate routed topology for the selected LFM and Qwen GGUF layouts."""
    arch = model.metadata.get("general.architecture")
    if arch not in ("lfm2moe", "qwen35moe"):
        return None

    experts = _integer(model, f"{arch}.expert_count")
    top_k = _integer(model, f"{arch}.expert_used_count")
    if top_k > experts:
        raise GGUFError(f"Router top-k {top_k} exceeds the expert count {experts}")
    blocks = _integer(model, f"{arch}.block_count")
    embedding = _integer(model, f"{arch}.embedding_length")
    inner = _integer(model, f"{arch}.expert_feed_forward_length")
    dense_blocks = 0
    if arch == "lfm2moe" and f"{arch}.leading_dense_block_count" in model.metadata:
        dense_blocks = _integer(model, f"{arch}.leading_dense_block_count", minimum=0)
    if not dense_blocks < blocks <= len(model.tensors):
        raise GGUFError(f"Invalid {arch} block counts (total={blocks}, dense={dense_blocks})")

    for suffix in ("expert_group_count", "expert_group_used_count"):
        key = f"{arch}.{suffix}"
        if key in model.metadata and _integer(model, key, minimum=0) > 1:
            raise GGUFError(f"Unsupported grouped routing metadata {key}")
    attention_experts = f"{arch}.attention.value_expert_count"
    if attention_experts in model.metadata and _integer(model, attention_experts, minimum=0):
        raise GGUFError(f"Unsupported attention routing metadata {attention_experts}")

    slots = [
        ("ffn_gate_inp.weight",        (embedding, experts),        1, "router", _F32),
        ("ffn_gate_exps.weight",       (embedding, inner, experts), 2, "gate", None),
        ("ffn_up_exps.weight",         (embedding, inner, experts), 2, "up", None),
        ("ffn_down_exps.weight",       (inner, embedding, experts), 2, "down", None),
    ]
    if arch == "lfm2moe":
        slots.append(("exp_probs_b.bias", (experts,), 0, "expert_bias", _F32))
        blocker = "Production LFM compaction requires an exact authorized donor and runtime qualification"
    else:
        shared = _integer(model, f"{arch}.expert_shared_feed_forward_length")
        blocker = "Production KAT compaction requires same-parent conversion and runtime qualification"

    layers = []
    expected = set()
    for i in range(dense_blocks, blocks):
        items = []
        for suffix, shape, axis, role, kind in slots:
            name = f"blk.{i}.{suffix}"
            _check_tensor(model, name, shape, kind)
            expected.add(name)
            items.append(ExpertTensor(name, axis, role))
        if arch == "qwen35moe":
            _check_tensor(model, f"blk.{i}.ffn_gate_inp_shexp.weight", (embedding,), _F32)
            _check_tensor(model, f"blk.{i}.ffn_gate_shexp.weight", (embedding, shared))
            _check_tensor(model, f"blk.{i}.ffn_up_shexp.weight", (embedding, shared))
            _check_tensor(model, f"blk.{i}.ffn_down_shexp.weight", (shared, embedding))
        layers.append(MoeLayer(i, tuple(items)))

    unknown = sorted(name for name in model.tensors if name not in expected and _ROUTED_NAME.match(name))
    if unknown:
        raise GGUFError(f"Unsupported or out-of-layer routed tensor {unknown[0]}")

    from .fixtures import FIXTURE_MARKER, check_fixture_layout

    can_compact = False
    if FIXTURE_MARKER in model.metadata:
        check_fixture_layout(model, experts)
        can_compact = True
    return MoeLayout(arch, f"{arch}.expert_count", experts, top_k, tuple(layers), can_compact,
                     None if can_compact else blocker)


def research_arms() -> dict:
    """Unrun research hypotheses, with published references identified separately."""
    return {
        "schema": "rsn.research-arms.v1",
        "kat": {
            "architecture": "qwen35moe",
            "source": "Kwaipilot/KAT-Coder-V2.5-Dev",
            "parent_experts": 256,
            "routed_layers": 40,
            "top_k": 8,
            "quantization_policy": "matched parent, conversion and Q4-compatible quantization",
            "production_compaction": "blocked pending donor, calibration and runtime qualification",
            "arms": [
                {"name": "Stock", "survivors": 256, "status": "proposed baseline"},
                {"name": "Greyhound-224", "survivors": 224, "status": "proposed"},
                {"name": "Greyhound-192", "survivors": 192, "status": "proposed"},
                {"name": "Greyhound-160", "survivors": 160, "status": "proposed"},
                {"name": "Published REAP-50", "survivors": 128, "status": "external reference",
                 "source": "Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF"},
            ],
        },
        "lfm": {
            "architecture": "lfm2moe",
            "source": "DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF",
            "parent_experts": 64,
            "routed_layers": 38,
            "top_k": 4,
            "quantization_policy": "pin exact NEO or MAX-NEO Q6_K donor SHA-256",
            "production_compaction": "blocked pending exact Q6_K donor and runtime qualification",
            "arms": [{"removed": count, "survivors": 64 - count, "status": "proposed"}
                     for count in (0, 8, 12, 16)],
        },
    }
