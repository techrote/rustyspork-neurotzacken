"""Resolve declarative edits to checked GGUF byte ranges."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

from .adapters import inspect_moe
from .gguf import GGUF, GGUFError, Tensor, align_up, copy_range, hash_range, layout_tensors, pad_to, write_header

PLAN_SCHEMA = "rsn.surgery-plan.v1"
MAX_INDICES = 4096
MAX_MAPPED_EXPERTS = 262_144
FLOATS = {0: struct.Struct("<f"), 1: struct.Struct("<e")}
OPERATIONS = frozenset({"probe", "scoop", "scrape", "gouge"})


@dataclass
class Budget:
    max_output_bytes: int = 64 * 1024 * 1024
    chunk_size: int = 1024 * 1024
    max_seconds: int = 60
    started: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        if self.max_output_bytes < 1 or not 4096 <= self.chunk_size <= 8 * 1024 * 1024:
            raise GGUFError("Output budget must be positive and chunks must be 4 KiB–8 MiB")
        if self.max_seconds < 1:
            raise GGUFError("Deadline must be at least one second")

    def check(self) -> None:
        if time.monotonic() - self.started >= self.max_seconds:
            raise GGUFError(f"Operation exceeded the {self.max_seconds}-second deadline")

    def check_output(self, size: int) -> None:
        if size > self.max_output_bytes:
            raise GGUFError(f"Candidate requires {size} bytes; output budget is {self.max_output_bytes}")

    def receipt(self) -> dict:
        return {
            "max_output_bytes": self.max_output_bytes,
            "io_chunk_bytes": self.chunk_size,
            "max_seconds": self.max_seconds,
            "device": "cpu",
            "network": "disabled",
        }


@dataclass(frozen=True)
class ExpertEdit:
    axis: int
    survivors: tuple[int, ...]
    count: int
    slice_bytes: int
    groups: int

    def ranges(self, offset: int) -> Iterator[tuple[int, int]]:
        for group in range(self.groups):
            base = offset + group * self.count * self.slice_bytes
            for survivor in self.survivors:
                yield base + survivor * self.slice_bytes, self.slice_bytes


def expert_edit(tensor: Tensor, axis: int, survivors: tuple[int, ...]) -> ExpertEdit:
    inner = math.prod(tensor.shape[:axis])
    if inner % tensor.block_elements:
        raise GGUFError(f"{tensor.name}: expert slices split {tensor.type_name} storage blocks")
    return ExpertEdit(
        axis, survivors, tensor.shape[axis], inner // tensor.block_elements * tensor.block_bytes,
        math.prod(tensor.shape[axis + 1:]),
    )


@dataclass(frozen=True)
class NumericEdit:
    name: str
    indices: tuple[int, ...]
    operation: str
    amount: float
    seed: int = 0

    def changes(self, source: BinaryIO, model: GGUF, budget: Budget) -> Iterator[tuple[int, bytes, dict]]:
        tensor = model.tensors[self.name]
        codec = FLOATS[tensor.ggml_type]
        base = model.data_offset + tensor.offset
        for index in self.indices:
            budget.check()
            offset = base + index * codec.size
            old_bytes = read_at(source, offset, codec.size)
            old = codec.unpack(old_bytes)[0]
            if not math.isfinite(old):
                raise GGUFError(f"{tensor.name}[{index}]: source value is not finite")
            if self.operation == "scrape" and self.amount == 0:
                modified = old
            elif self.operation == "scoop":
                modified = old * self.amount
            else:
                noise = hashlib.sha256(struct.pack("<QQ", self.seed, index)).digest()
                unit = (int.from_bytes(noise[:8], "little") >> 11) / (1 << 53)
                modified = old + self.amount * (2 * unit - 1)
            try:
                output = codec.pack(modified)
            except (OverflowError, struct.error) as error:
                raise GGUFError(f"{tensor.name}[{index}]: edit overflows {tensor.type_name}") from error
            new = codec.unpack(output)[0]
            if not math.isfinite(new):
                raise GGUFError(f"{tensor.name}[{index}]: edit produces a non-finite value")
            yield offset, output, {
                "index": index, "before": old, "after": new, "bytes_changed": old_bytes != output,
            }


@dataclass
class Prepared:
    model: GGUF
    plan: dict
    kind: str
    source_sha256: str
    tensors: tuple[Tensor, ...]
    candidate_bytes: int
    header_bytes: int
    experts: dict[str, ExpertEdit] = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    mappings: list[dict] = field(default_factory=list)
    compaction_blocker: str | None = None
    numeric: NumericEdit | None = None
    numeric_changes: tuple[tuple[int, bytes, dict], ...] = ()
    probe: dict | None = None

    def summary(self) -> dict:
        return {
            "schema_version": "rsn.dry-run.v1",
            "state": "validated",
            "operation": self.kind,
            "source_sha256": self.source_sha256,
            "source_file_bytes": self.model.size,
            "predicted_candidate_bytes": self.candidate_bytes if self.kind != "probe" else None,
            "predicted_file_bytes_saved": self.model.size - self.candidate_bytes if self.kind != "probe" else None,
            "predicted_expert_tensor_bytes_removed": sum(
                self.model.tensors[tensor.name].nbytes - tensor.nbytes for tensor in self.tensors
            ),
            "metadata_overrides": self.overrides,
            "compaction_blocker": self.compaction_blocker,
            "expert_mappings": self.mappings,
            "structural_tensor_changes": [
                {"name": name, "shape_before": list(self.model.tensors[name].shape),
                 "shape_after": list(tensor.shape), "bytes_before": self.model.tensors[name].nbytes,
                 "bytes_after": tensor.nbytes, "expert_axis": self.experts[name].axis}
                for tensor in self.tensors if (name := tensor.name) in self.experts
            ],
            "numeric_selection": {"tensor": self.numeric.name, "indices": list(self.numeric.indices)}
            if self.numeric else None,
        }


class DigestWriter:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.size = 0

    def tell(self) -> int:
        return self.size

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size += len(data)
        return len(data)


def sha256(model: GGUF, budget: Budget, source: BinaryIO | None = None) -> str:
    return model.sha256(source_file=source, chunk_size=budget.chunk_size, check=budget.check)


def read_at(source: BinaryIO, offset: int, size: int) -> bytes:
    source.seek(offset)
    data = source.read(size)
    if len(data) != size:
        raise GGUFError("Unexpected end of checkpoint during tensor read")
    return data


def fields(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise GGUFError(f"{label} must contain exactly these fields: {', '.join(sorted(expected))}")


def indices(value: object, maximum: int, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INDICES:
        raise GGUFError(f"{label} must be a nonempty list of up to {MAX_INDICES} indices")
    if any(type(item) is not int or not 0 <= item < maximum for item in value):
        raise GGUFError(f"{label} contains an invalid index; expected 0–{maximum - 1}")
    if any(left >= right for left, right in zip(value, value[1:])):
        raise GGUFError(f"{label} must be unique and in ascending source order")
    return tuple(value)


def tensor_identity(tensor: Tensor) -> dict:
    return {"name": tensor.name, "shape": list(tensor.shape), "ggml_type": tensor.ggml_type}


def make_plan(
    model: GGUF, budget: Budget, *, operation: str, keep: list[int] | None = None,
    tensor_name: str | None = None, selected: list[int] | None = None,
    scale: float | None = None, amplitude: float | None = None, seed: int | None = None,
) -> dict:
    if operation == "gouge":
        layout = inspect_moe(model)
        if layout is None:
            raise GGUFError("Source has no supported MoE layout")
        if tensor_name is not None or selected is not None or any(v is not None for v in (scale, amplitude, seed)):
            raise GGUFError("Gouge accepts expert IDs through --keep")
        payload = {"family": operation, "layers": [
            {"layer": layer.index, "survivors": list(keep) if keep is not None else None} for layer in layout.layers
        ]}
    elif operation in OPERATIONS:
        if keep is not None or tensor_name not in model.tensors:
            raise GGUFError("An exact source --tensor name and --indices are required")
        if (operation == "probe" and any(v is not None for v in (scale, amplitude, seed))
                or operation == "scoop" and (amplitude is not None or seed is not None)
                or operation == "scrape" and scale is not None):
            raise GGUFError("Numeric parameters do not match the selected operation")
        payload = {"family": operation, "tensor": tensor_identity(model.tensors[tensor_name]), "indices": selected}
        if operation == "scoop":
            payload["scale"] = scale
        elif operation == "scrape":
            payload.update(amplitude=amplitude, seed=seed)
    else:
        raise GGUFError(f"Operation {operation!r} is unsupported by this engine")
    source_hash = sha256(model, budget)
    plan = {"schema_version": PLAN_SCHEMA, "source_sha256": source_hash, "operation": payload}
    prepare(model, plan, budget, source_hash=source_hash, enforce_output=False, require_compaction=False)
    return plan


def prepare(
    model: GGUF, plan: dict, budget: Budget, *, source_hash: str | None = None,
    enforce_output: bool = True, require_compaction: bool = True,
) -> Prepared:
    fields(plan, {"schema_version", "source_sha256", "operation"}, "Plan")
    if plan["schema_version"] != PLAN_SCHEMA:
        raise GGUFError(f"Expected plan schema {PLAN_SCHEMA}")
    claimed_hash = plan["source_sha256"]
    if not isinstance(claimed_hash, str) or len(claimed_hash) != 64 or any(
        char not in "0123456789abcdef" for char in claimed_hash
    ):
        raise GGUFError("Plan requires a lowercase SHA-256 donor checksum")
    operation = plan["operation"]
    kind = operation.get("family") if isinstance(operation, dict) else None
    if not isinstance(kind, str) or kind not in OPERATIONS:
        raise GGUFError(f"Operation {kind!r} has no implemented backend")
    if model.metadata.get("split.count", 1) != 1:
        raise GGUFError("Surgery requires a complete single-file donor")
    with model.open() as source:
        source_hash = source_hash or sha256(model, budget, source)
        if claimed_hash != source_hash:
            raise GGUFError("Plan donor SHA-256 does not match the selected source")
        result = Prepared(model, plan, kind, source_hash, tuple(model.tensors.values()), model.size, model.data_offset)
        if kind == "gouge":
            resolve_experts(result, operation, source, budget)
        else:
            resolve_numeric(result, operation, source, budget)
        model.assert_unchanged(source)
    if require_compaction and result.compaction_blocker:
        raise GGUFError(result.compaction_blocker)
    if enforce_output and kind != "probe":
        budget.check_output(result.candidate_bytes)
    budget.check()
    return result


def resolve_experts(result: Prepared, operation: dict, source: BinaryIO, budget: Budget) -> None:
    fields(operation, {"family", "layers"}, "Gouge operation")
    layout = inspect_moe(result.model)
    if layout is None:
        raise GGUFError("Source has no supported MoE layout")
    layers = operation["layers"]
    if not isinstance(layers, list) or len(layers) != len(layout.layers):
        raise GGUFError("Plan must name every routed layer exactly once in source order")
    if layout.expert_count > MAX_INDICES or len(layers) * layout.expert_count > MAX_MAPPED_EXPERTS:
        raise GGUFError("Source exceeds the bounded expert manifest size")
    counts = set()
    specs = {tensor.name: tensor.shape for tensor in result.tensors}
    for target, layer in zip(layers, layout.layers):
        budget.check()
        fields(target, {"layer", "survivors"}, "Layer selection")
        if type(target["layer"]) is not int or target["layer"] != layer.index:
            raise GGUFError(f"Expected routed layer {layer.index}")
        survivors = indices(target["survivors"], layout.expert_count, f"Layer {layer.index} survivors")
        count = len(survivors)
        if count < layout.top_k:
            raise GGUFError(f"Layer {layer.index}: {count} survivors cannot satisfy top-{layout.top_k} routing")
        counts.add(count)
        mapping = [None] * layout.expert_count
        for new, old in enumerate(survivors):
            mapping[old] = new
        result.mappings.append({"layer": layer.index, "survivors": list(survivors), "old_to_new": mapping})
        if count == layout.expert_count:
            continue
        for selection in layer.tensors:
            tensor = result.model.tensors[selection.name]
            axis = selection.axis
            result.experts[tensor.name] = expert_edit(tensor, axis, survivors)
            specs[tensor.name] = tensor.shape[:axis] + (count,) + tensor.shape[axis + 1:]
    if len(counts) != 1:
        raise GGUFError("This GGUF architecture requires one surviving expert count across all routed layers")
    if not result.experts:
        return
    if not layout.can_compact:
        result.compaction_blocker = layout.blocker or "Physical compaction is not qualified for this donor"
    validate_router_values(result, source, budget)
    result.overrides = {layout.expert_count_key: counts.pop()}
    result.tensors = layout_tensors(
        [(tensor.name, specs[tensor.name], tensor.ggml_type) for tensor in result.tensors], result.model.alignment,
    )
    result.header_bytes = write_header(
        DigestWriter(), result.model, result.tensors, result.overrides,
        source_file=source, chunk_size=budget.chunk_size, check=budget.check,
    )
    last_tensor_end = max(t.offset + t.nbytes for t in result.tensors)
    result.candidate_bytes = result.header_bytes + align_up(last_tensor_end, result.model.alignment)


def validate_router_values(result: Prepared, source: BinaryIO, budget: Budget) -> None:
    for name in result.experts:
        if not name.endswith((".ffn_gate_inp.weight", ".exp_probs_b.bias")):
            continue
        tensor = result.model.tensors[name]
        offset = result.model.data_offset + tensor.offset
        codec = FLOATS.get(tensor.ggml_type)
        if codec is None:
            raise GGUFError(f"{name}: router and bias require floating point storage")
        block_bytes = max(1, budget.chunk_size // codec.size) * codec.size
        for start in range(0, tensor.nbytes, block_bytes):
            budget.check()
            data = read_at(source, offset + start, min(block_bytes, tensor.nbytes - start))
            if any(not math.isfinite(value) for value, in codec.iter_unpack(data)):
                raise GGUFError(f"{name}: router or bias contains a non-finite value")


def resolve_numeric(result: Prepared, operation: dict, source: BinaryIO, budget: Budget) -> None:
    expected = {"family", "tensor", "indices"}
    if result.kind == "scoop":
        expected.add("scale")
    elif result.kind == "scrape":
        expected.update(("amplitude", "seed"))
    fields(operation, expected, "Numeric operation")
    target = operation["tensor"]
    fields(target, {"name", "shape", "ggml_type"}, "Tensor identity")
    name = target["name"]
    tensor = result.model.tensors.get(name) if isinstance(name, str) else None
    if (tensor is None or type(target["ggml_type"]) is not int or not isinstance(target["shape"], list)
            or any(type(dim) is not int for dim in target["shape"]) or target != tensor_identity(tensor)):
        raise GGUFError("Tensor identity does not match the exact source inventory")
    selected = indices(operation["indices"], tensor.logical_elements, "Logical flat indices")
    if result.kind == "probe":
        result.probe = probe_tensor(result, source, tensor, selected, budget)
        return
    if tensor.ggml_type not in FLOATS:
        raise GGUFError("Numeric edits support F32/F16; quantized edits require a verified re-encoder")
    amount_key = "scale" if result.kind == "scoop" else "amplitude"
    amount = operation[amount_key]
    if type(amount) not in (float, int) or not 0 <= amount <= 1:
        raise GGUFError(f"{amount_key} must be a number in [0, 1]")
    seed = operation.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise GGUFError("Scrape requires an unsigned 64-bit seed")
    result.numeric = NumericEdit(name, selected, result.kind, amount, seed)
    result.numeric_changes = tuple(result.numeric.changes(source, result.model, budget))


def q6_value(block: bytes, index: int) -> float:
    """Decode one ggml Q6_K logical value from its 210-byte superblock."""
    # Field order and lane groups follow llama.cpp's dequantize_row_q6_K.
    half, within = divmod(index, 128)
    group, lane = divmod(within, 32)
    low = block[half * 64 + group % 2 * 32 + lane]
    low = (low >> (4 * (group // 2))) & 15
    high = (block[128 + half * 32 + lane] >> (group * 2)) & 3
    scale = struct.unpack_from("<b", block, 192 + half * 8 + group * 2 + lane // 16)[0]
    delta = struct.unpack_from("<e", block, 208)[0]
    return delta * scale * (low + 16 * high - 32)


def probe_tensor(result: Prepared, source: BinaryIO, tensor: Tensor, selected: tuple[int, ...], budget: Budget) -> dict:
    base = result.model.data_offset + tensor.offset
    codec = FLOATS.get(tensor.ggml_type)
    if codec is None and tensor.type_name != "Q6_K":
        raise GGUFError(f"Probe cannot decode {tensor.type_name}; run inventory for its storage geometry")
    samples = []
    block_number, block = -1, b""
    for index in selected:
        budget.check()
        if codec is not None:
            value = codec.unpack(read_at(source, base + index * codec.size, codec.size))[0]
        else:
            number, lane = divmod(index, tensor.block_elements)
            if number != block_number:
                block = read_at(source, base + number * tensor.block_bytes, tensor.block_bytes)
                block_number = number
            value = q6_value(block, lane)
        samples.append({"index": index, "value": value if math.isfinite(value) else None})
    values = [record["value"] for record in samples if record["value"] is not None]
    return {
        "schema_version": "rsn.tensor-probe.v1", "evidence": "measured_selected_logical_weights",
        "source_sha256": result.source_sha256, "tensor": tensor_identity(tensor),
        "coordinate_order": "GGML axis 0 fastest", "samples": samples,
        "coverage": {"sampled": len(samples), "logical_elements": tensor.logical_elements},
        "non_finite": len(samples) - len(values),
        "statistics": {"min": min(values), "max": max(values), "mean": math.fsum(values) / len(values)}
        if values else None,
    }


def write_candidate(out: BinaryIO, source: BinaryIO, prepared: Prepared, budget: Budget) -> None:
    model = prepared.model
    if prepared.experts:
        write_header(out, model, prepared.tensors, prepared.overrides, source_file=source,
                     chunk_size=budget.chunk_size, check=budget.check)
        for tensor in prepared.tensors:
            pad_to(out, prepared.header_bytes + tensor.offset, chunk_size=budget.chunk_size, check=budget.check)
            original = model.tensors[tensor.name]
            start = model.data_offset + original.offset
            edit = prepared.experts.get(tensor.name)
            ranges = edit.ranges(start) if edit else [(start, original.nbytes)]
            for offset, length in ranges:
                copy_range(source, out, offset, length, chunk_size=budget.chunk_size, check=budget.check)
        pad_to(out, prepared.candidate_bytes, chunk_size=budget.chunk_size, check=budget.check)
        return
    cursor = 0
    for offset, value, _ in prepared.numeric_changes:
        copy_range(source, out, cursor, offset - cursor, chunk_size=budget.chunk_size, check=budget.check)
        out.write(value)
        cursor = offset + len(value)
    copy_range(source, out, cursor, model.size - cursor, chunk_size=budget.chunk_size, check=budget.check)


def verify_padding(stream: BinaryIO, start: int, end: int, budget: Budget, label: str) -> None:
    for offset in range(start, end, budget.chunk_size):
        budget.check()
        if any(read_at(stream, offset, min(end - offset, budget.chunk_size))):
            raise GGUFError(f"{label}: candidate padding is not zero")


def compare_ranges(
    source: BinaryIO, candidate: BinaryIO, start: int, target: int, length: int, budget: Budget, label: str,
) -> None:
    for offset in range(0, length, budget.chunk_size):
        budget.check()
        size = min(budget.chunk_size, length - offset)
        if read_at(source, start + offset, size) != read_at(candidate, target + offset, size):
            raise GGUFError(f"{label}: candidate bytes differ at file offset {target + offset}")


def verify_candidate(prepared: Prepared, candidate: GGUF, budget: Budget) -> dict:
    model = prepared.model
    if prepared.kind == "probe":
        raise GGUFError("Probe has no candidate checkpoint")
    if candidate.size != prepared.candidate_bytes:
        raise GGUFError(f"Candidate file size mismatch: expected {prepared.candidate_bytes}, found {candidate.size}")
    if tuple(candidate.tensors.values()) != prepared.tensors:
        raise GGUFError("Candidate tensor descriptors differ from the validated plan")
    changes = []
    tensors = []
    with model.open() as source, candidate.open() as target:
        if prepared.experts:
            expected = DigestWriter()
            write_header(expected, model, prepared.tensors, prepared.overrides, source_file=source,
                         chunk_size=budget.chunk_size, check=budget.check)
            header = hash_range(target, 0, candidate.data_offset, chunk_size=budget.chunk_size, check=budget.check)
            if candidate.data_offset != expected.size or header != expected.digest.hexdigest():
                raise GGUFError("Candidate header differs from exact metadata and tensor layout")
            cursor = candidate.data_offset
            for tensor in prepared.tensors:
                original = model.tensors[tensor.name]
                start = model.data_offset + original.offset
                output = candidate.data_offset + tensor.offset
                verify_padding(target, cursor, output, budget, tensor.name)
                edit = prepared.experts.get(tensor.name)
                ranges = edit.ranges(start) if edit else [(start, original.nbytes)]
                for offset, length in ranges:
                    compare_ranges(source, target, offset, output, length, budget, tensor.name)
                    output += length
                cursor = output
            verify_padding(target, cursor, candidate.size, budget, "End of file")
        else:
            cursor = 0
            for offset, value, observation in prepared.numeric_changes:
                compare_ranges(source, target, cursor, cursor, offset - cursor, budget, "Unselected source")
                if read_at(target, offset, len(value)) != value:
                    raise GGUFError(f"Candidate numeric edit differs at logical index {observation['index']}")
                changes.append(observation)
                cursor = offset + len(value)
            compare_ranges(source, target, cursor, cursor, model.size - cursor, budget, "Unselected source")
        for tensor in prepared.tensors:
            original = model.tensors[tensor.name]
            old_hash = hash_range(source, model.data_offset + original.offset, original.nbytes,
                                  chunk_size=budget.chunk_size, check=budget.check)
            new_hash = hash_range(target, candidate.data_offset + tensor.offset, tensor.nbytes,
                                  chunk_size=budget.chunk_size, check=budget.check)
            tensors.append({
                "name": tensor.name, "ggml_type": tensor.ggml_type,
                "source_shape": list(original.shape), "candidate_shape": list(tensor.shape),
                "source_bytes": original.nbytes, "candidate_bytes": tensor.nbytes,
                "source_sha256": old_hash, "candidate_sha256": new_hash,
                "changed": original.shape != tensor.shape or old_hash != new_hash,
            })
        candidate_hash = sha256(candidate, budget, target)
        if sha256(model, budget, source) != prepared.source_sha256:
            raise GGUFError("Donor SHA-256 changed during candidate verification")
        model.assert_unchanged(source)
        candidate.assert_unchanged(target)
    return {
        "schema_version": "rsn.surgery-verification.v1", "state": "mechanically_verified",
        "evidence": "measured_checkpoint_bytes", "source_sha256": prepared.source_sha256,
        "candidate_sha256": candidate_hash, "source_file_bytes": model.size,
        "candidate_file_bytes": candidate.size, "file_bytes_saved": model.size - candidate.size,
        "tensors": tensors, "expert_mappings": prepared.mappings, "metadata_overrides": prepared.overrides,
        "numeric_tensor": prepared.numeric.name if prepared.numeric else None,
        "numeric_observations": changes,
    }
