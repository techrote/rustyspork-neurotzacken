"""Bounded little-endian GGUF v2/v3 inspection and serialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO


class GGUFError(ValueError):
    """Malformed data or an unsupported GGUF feature."""


DEFAULT_ALIGNMENT = 32
DEFAULT_CHUNK = 1024 * 1024
QUANTIZATION_VERSION = 2
MAX_ALIGNMENT = 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 1024 * 1024
MAX_RECORDS = 100_000
MAX_ARRAY_ELEMENTS = 8_000_000
MAX_I64 = (1 << 63) - 1

# Stable on-disk types from GGMLQuantizationType and GGML_QUANT_SIZES:
# https://github.com/ggml-org/llama.cpp/blob/1f8baae28d096624bd47d33d2d179ec479d47720/gguf-py/gguf/constants.py
# Each entry is (name, logical elements per block, bytes per block).
TYPE_SIZES = MappingProxyType({
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
    43: ("F8_E4M3", 1, 1),
})

UINT32 = 4
BOOL = 7
STRING = 8
ARRAY = 9
_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
    7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}
_VALUE_TYPES = (
    "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "FLOAT32",
    "BOOL", "STRING", "ARRAY", "UINT64", "INT64", "FLOAT64",
)
_METADATA_KEY = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")
_PREFIX = struct.Struct("<4sIQQ")
Check = Callable[[], None]


@dataclass(frozen=True)
class ArrayInfo:
    element_type: int
    count: int


@dataclass(frozen=True)
class MetadataRecord:
    type_id: int
    offset: int
    nbytes: int


MetadataValue = int | float | bool | str | ArrayInfo


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int

    @property
    def type_name(self) -> str:
        return TYPE_SIZES[self.ggml_type][0]

    @property
    def block_elements(self) -> int:
        return TYPE_SIZES[self.ggml_type][1]

    @property
    def block_bytes(self) -> int:
        return TYPE_SIZES[self.ggml_type][2]

    @property
    def logical_elements(self) -> int:
        return math.prod(self.shape)

    @property
    def row_bytes(self) -> int:
        return self.shape[0] // self.block_elements * self.block_bytes


def _identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev, file_stat.st_ino, file_stat.st_size,
        file_stat.st_mtime_ns, file_stat.st_ctime_ns,
    )


@dataclass(frozen=True)
class GGUF:
    path: Path
    size: int
    version: int
    alignment: int
    data_offset: int
    metadata: Mapping[str, MetadataValue]
    metadata_records: Mapping[str, MetadataRecord]
    tensors: Mapping[str, Tensor]
    _identity: tuple[int, int, int, int, int]
    _path_identity: tuple[int, int, int, int, int]

    def assert_unchanged(self, source_file: BinaryIO | None = None) -> None:
        try:
            current = _identity(self.path.stat())
            opened = _identity(os.fstat(source_file.fileno())) if source_file else self._identity
        except FileNotFoundError as error:
            raise GGUFError(f"GGUF source disappeared: {self.path}") from error
        if self._path_identity != current or self._identity != opened:
            raise GGUFError(f"GGUF source changed since inspection: {self.path}")

    @contextmanager
    def open(self) -> Iterator[BinaryIO]:
        """Open a read-only descriptor checked against the inspected source identity."""
        with self.path.open("rb") as source_file:
            self.assert_unchanged(source_file)
            yield source_file
            self.assert_unchanged(source_file)

    def sha256(
        self, *, source_file: BinaryIO | None = None,
        chunk_size: int = DEFAULT_CHUNK, check: Check | None = None,
    ) -> str:
        with self.open() if source_file is None else nullcontext(source_file) as stream:
            self.assert_unchanged(stream)
            digest = hash_range(stream, 0, self.size, chunk_size, check=check)
            self.assert_unchanged(stream)
        return digest

    def metadata_array(
        self, key: str, *, max_elements: int = 65_536, source_file: BinaryIO | None = None,
        check: Check | None = None,
    ) -> tuple[MetadataValue, ...]:
        """Read a selected metadata array with an explicit element bound."""
        info = self.metadata.get(key)
        if not isinstance(info, ArrayInfo):
            raise GGUFError(f"GGUF metadata key {key!r} is not an array")
        if type(max_elements) is not int or not 0 <= info.count <= max_elements:
            raise GGUFError(f"GGUF metadata array {key!r} exceeds the requested element limit")
        record = self.metadata_records[key]
        with self.open() if source_file is None else nullcontext(source_file) as stream:
            self.assert_unchanged(stream)
            stream.seek(record.offset + 12 + len(key))
            reader = _Reader(stream, record.offset + record.nbytes, check)
            element_type, count = reader.unpack("<IQ")
            if (element_type, count) != (info.element_type, info.count):
                raise GGUFError(f"GGUF metadata array {key!r} changed since inspection")
            elements = tuple(reader.value(element_type) for _ in range(count))
            self.assert_unchanged(stream)
        return elements

    def inventory(
        self, *, include_sha256: bool = True, source_sha256: str | None = None,
        source_file: BinaryIO | None = None, chunk_size: int = DEFAULT_CHUNK,
        check: Check | None = None,
    ) -> dict:
        """Return path-independent exact layout and an optional whole-file identity."""
        tensors = [
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "ggml_type": tensor.ggml_type,
                "type_name": tensor.type_name,
                "tensor_data_offset_bytes": tensor.offset,
                "file_offset_bytes": self.data_offset + tensor.offset,
                "storage_bytes": tensor.nbytes,
                "logical_elements": tensor.logical_elements,
                "block_elements": tensor.block_elements,
                "block_bytes": tensor.block_bytes,
                "row_bytes": tensor.row_bytes,
            }
            for tensor in self.tensors.values()
        ]
        tensor_json = json.dumps(tensors, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        metadata_size = sum(record.nbytes for record in self.metadata_records.values())
        with self.open() if source_file is None else nullcontext(source_file) as stream:
            self.assert_unchanged(stream)
            metadata_sha256 = hash_range(
                stream, _PREFIX.size, metadata_size, chunk_size, check=check,
            )
            if include_sha256 and source_sha256 is None:
                source_sha256 = self.sha256(source_file=stream, chunk_size=chunk_size, check=check)
            self.assert_unchanged(stream)
        return {
            "schema_version": "rsn.gguf-inventory.v1",
            "format": "GGUF",
            "version": self.version,
            "byte_order": "little",
            "alignment_bytes": self.alignment,
            "data_offset_bytes": self.data_offset,
            "size_bytes": self.size,
            "source_sha256": source_sha256,
            "metadata_sha256": metadata_sha256,
            "tensor_inventory_sha256": hashlib.sha256(tensor_json.encode("ascii")).hexdigest(),
            "tensor_axis_order": "axis_0_fastest",
            "architecture": self.metadata.get("general.architecture"),
            "metadata": {key: _json_metadata(value) for key, value in self.metadata.items()},
            "tensors": tensors,
        }


def _json_metadata(value: MetadataValue) -> object:
    if isinstance(value, ArrayInfo):
        return {"element_type": _VALUE_TYPES[value.element_type], "count": value.count}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def align_up(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _validate_alignment(alignment: object) -> int:
    if (
        type(alignment) is not int or not 1 <= alignment <= MAX_ALIGNMENT
        or alignment & (alignment - 1)
    ):
        raise GGUFError(f"GGUF alignment must be a power of two from 1 to {MAX_ALIGNMENT}")
    return alignment


def _tensor(name: str, shape: tuple[int, ...], ggml_type: int, offset: int) -> Tensor:
    if not name or "\x00" in name or len(name.encode("utf-8")) > 64:
        raise GGUFError("GGUF tensor names must contain 1 to 64 UTF-8 bytes without NUL")
    if not 1 <= len(shape) <= 4 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise GGUFError(f"Invalid shape for tensor {name!r}: {shape}")
    if type(ggml_type) is not int or ggml_type not in TYPE_SIZES:
        raise GGUFError(f"Unsupported GGML type {ggml_type} for tensor {name!r}")
    type_name, block_elements, block_bytes = TYPE_SIZES[ggml_type]
    logical_elements = math.prod(shape)
    if shape[0] % block_elements:
        raise GGUFError(
            f"Tensor {name!r} row has {shape[0]} elements, "
            f"not a multiple of the {type_name} block width {block_elements}"
        )
    nbytes = logical_elements // block_elements * block_bytes
    if logical_elements > MAX_I64 or offset + nbytes > MAX_I64:
        raise GGUFError(f"Tensor {name!r} exceeds signed 64-bit indexing bounds")
    return Tensor(name, shape, ggml_type, offset, nbytes)


def layout_tensors(
    tensors: Iterable[tuple[str, tuple[int, ...], int]], alignment: int = DEFAULT_ALIGNMENT,
) -> tuple[Tensor, ...]:
    """Compute llama.cpp-compatible relative offsets from physical tensor shapes."""
    _validate_alignment(alignment)
    offset = 0
    result = []
    names = set()
    for name, shape, ggml_type in tensors:
        if name in names:
            raise GGUFError(f"Duplicate tensor name: {name!r}")
        names.add(name)
        tensor = _tensor(name, tuple(shape), ggml_type, offset)
        result.append(tensor)
        offset = align_up(offset + tensor.nbytes, alignment)
    return tuple(result)


def read_exact(stream: BinaryIO, nbytes: int) -> bytes:
    data = stream.read(nbytes)
    if len(data) != nbytes:
        raise GGUFError(f"Truncated GGUF at file offset {stream.tell() - len(data)}")
    return data


class _Reader:
    def __init__(self, stream: BinaryIO, size: int, check: Check | None) -> None:
        self.stream = stream
        self.limit = min(size, MAX_HEADER_BYTES)
        self.check = check

    def _check_length(self, length: int) -> None:
        if length > self.limit - self.stream.tell():
            message = "Truncated GGUF header"
            if self.limit == MAX_HEADER_BYTES:
                message = "GGUF header exceeds inspection limit"
            raise GGUFError(f"{message} at file offset {self.stream.tell()}")
        if self.check:
            self.check()

    def read(self, length: int) -> bytes:
        self._check_length(length)
        return read_exact(self.stream, length)

    def skip(self, length: int) -> None:
        self._check_length(length)
        self.stream.seek(length, os.SEEK_CUR)

    def unpack(self, fmt: str) -> tuple:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))

    def string(self, limit: int = MAX_STRING_BYTES) -> str:
        (length,) = self.unpack("<Q")
        if length > limit:
            raise GGUFError(f"GGUF string exceeds {limit} bytes at file offset {self.stream.tell()}")
        try:
            return self.read(length).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise GGUFError(f"Invalid GGUF UTF-8 at file offset {self.stream.tell() - length}") from error

    def value(self, type_id: int) -> MetadataValue:
        if type_id == STRING:
            return self.string()
        if type_id == ARRAY:
            return self.array()
        if type_id not in _SCALAR_FORMATS:
            raise GGUFError(f"Unknown GGUF metadata type {type_id}")
        (value,) = self.unpack(_SCALAR_FORMATS[type_id])
        if type_id == BOOL:
            if value not in (0, 1):
                raise GGUFError("GGUF BOOL must be encoded as 0 or 1")
            return bool(value)
        return value

    def array(self) -> ArrayInfo:
        element_type, count = self.unpack("<IQ")
        if element_type == ARRAY:
            raise GGUFError("Nested GGUF arrays are unsupported by the target loader")
        if element_type >= len(_VALUE_TYPES):
            raise GGUFError(f"Unknown GGUF array element type {element_type}")
        if count > MAX_ARRAY_ELEMENTS:
            raise GGUFError(f"GGUF array exceeds {MAX_ARRAY_ELEMENTS} elements")
        if element_type == STRING:
            self._check_length(count * 8)
            for _ in range(count):
                self.string()
        elif element_type == BOOL:
            self._check_length(count)
            for start in range(0, count, DEFAULT_CHUNK):
                data = self.read(min(DEFAULT_CHUNK, count - start))
                if data.translate(None, b"\x00\x01"):
                    raise GGUFError("GGUF BOOL array must contain only encoded 0 or 1")
        else:
            self.skip(struct.calcsize(_SCALAR_FORMATS[element_type]) * count)
        return ArrayInfo(element_type, count)


def _metadata_key(key: str) -> None:
    if not _METADATA_KEY.fullmatch(key) or len(key) > 65535:
        raise GGUFError(f"Invalid GGUF metadata key: {key[:128]!r}")


def _read_metadata(reader: _Reader, count: int) -> tuple[dict, dict]:
    metadata = {}
    records = {}
    for _ in range(count):
        offset = reader.stream.tell()
        key = reader.string(65535)
        _metadata_key(key)
        if key in metadata:
            raise GGUFError(f"Duplicate GGUF metadata key: {key!r}")
        (type_id,) = reader.unpack("<I")
        metadata[key] = reader.value(type_id)
        records[key] = MetadataRecord(type_id, offset, reader.stream.tell() - offset)
    return metadata, records


def open_gguf(path: str | os.PathLike, *, check: Check | None = None) -> GGUF:
    """Inspect only the bounded GGUF header and derive exact tensor storage spans."""
    path = Path(path).resolve(strict=True)
    path_info = path.stat()
    if not stat.S_ISREG(path_info.st_mode):
        raise GGUFError(f"Expected a regular GGUF file: {path}")
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        # Windows path stat and descriptor stat may report different ctime fields.
        path_identity, identity = _identity(path_info), _identity(info)
        if path_identity[:4] != identity[:4]:
            raise GGUFError(f"GGUF source changed while opening: {path}")
        reader = _Reader(stream, info.st_size, check)
        magic, version, tensor_count, metadata_count = _PREFIX.unpack(reader.read(_PREFIX.size))
        if magic != b"GGUF":
            raise GGUFError("Expected the GGUF file signature")
        if version in (2 << 24, 3 << 24):
            raise GGUFError("Big-endian GGUF is unsupported")
        if version not in (2, 3):
            raise GGUFError(f"Unsupported GGUF version {version}. Expected version 2 or 3")
        if max(tensor_count, metadata_count) > MAX_RECORDS:
            raise GGUFError(f"GGUF count exceeds {MAX_RECORDS} tensor or metadata records")
        reader._check_length(tensor_count * 33 + metadata_count * 14)
        metadata, records = _read_metadata(reader, metadata_count)
        if "general.alignment" in records and records["general.alignment"].type_id != UINT32:
            raise GGUFError("GGUF general.alignment must use UINT32")
        alignment = _validate_alignment(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        architecture = metadata.get("general.architecture")
        if "general.architecture" in records and (not isinstance(architecture, str) or not architecture):
            raise GGUFError("GGUF general.architecture must be a nonempty string")
        tensors = {}
        next_offset = end = 0
        for _ in range(tensor_count):
            name = reader.string(64)
            if name in tensors:
                raise GGUFError(f"Duplicate tensor name: {name!r}")
            (ndim,) = reader.unpack("<I")
            if not 1 <= ndim <= 4:
                raise GGUFError(f"Invalid rank {ndim} for tensor {name!r}. Expected 1 to 4")
            shape = reader.unpack(f"<{ndim}Q")
            ggml_type, offset = reader.unpack("<IQ")
            if offset != next_offset:
                raise GGUFError(
                    f"Tensor {name!r} has byte offset {offset}, expected packed aligned offset {next_offset}"
                )
            tensor = _tensor(name, shape, ggml_type, offset)
            end = offset + tensor.nbytes
            next_offset = align_up(end, alignment)
            tensors[name] = tensor
        quantization_key = "general.quantization_version"
        quantization_record = records.get(quantization_key)
        if quantization_record is None:
            if any(tensor.block_elements > 1 for tensor in tensors.values()):
                raise GGUFError("Quantized GGUF requires general.quantization_version")
        else:
            if quantization_record.type_id != UINT32:
                raise GGUFError("GGUF general.quantization_version must use UINT32")
            if metadata[quantization_key] != QUANTIZATION_VERSION:
                raise GGUFError(
                    f"Unsupported GGML quantization version {metadata[quantization_key]}. "
                    f"Expected {QUANTIZATION_VERSION}"
                )
        header_end = stream.tell()
        data_offset = align_up(header_end, alignment) if tensors else header_end
        if data_offset + end > info.st_size:
            raise GGUFError("GGUF tensor payload extends beyond the end of the file")
        max_size = data_offset + next_offset if tensors else align_up(header_end, alignment)
        if info.st_size > max_size:
            raise GGUFError("Unexpected trailing bytes after the GGUF tensor data")
        if tensors and reader.read(data_offset - header_end).strip(b"\x00"):
            raise GGUFError("Nonzero GGUF header alignment padding")
        result = GGUF(
            path, info.st_size, version, alignment, data_offset,
            MappingProxyType(metadata), MappingProxyType(records), MappingProxyType(tensors),
            identity, path_identity,
        )
        result.assert_unchanged(stream)
        return result


def _range_chunks(
    stream: BinaryIO, offset: int, length: int, chunk_size: int, check: Check | None,
) -> Iterator[bytes]:
    if offset < 0 or length < 0 or type(chunk_size) is not int or chunk_size <= 0:
        raise GGUFError("Copy or hash requires a nonnegative byte range and positive chunk size")
    stream.seek(offset)
    remaining = length
    while remaining:
        if check:
            check()
        block = read_exact(stream, min(remaining, chunk_size))
        yield block
        remaining -= len(block)
    if check:
        check()


def hash_range(
    stream: BinaryIO, offset: int, length: int, chunk_size: int = DEFAULT_CHUNK,
    *, check: Check | None = None,
) -> str:
    digest = hashlib.sha256()
    for block in _range_chunks(stream, offset, length, chunk_size, check):
        digest.update(block)
    return digest.hexdigest()


def copy_range(
    source: BinaryIO, out: BinaryIO, offset: int, length: int, chunk_size: int = DEFAULT_CHUNK,
    *, check: Check | None = None,
) -> None:
    for block in _range_chunks(source, offset, length, chunk_size, check):
        _write(out, block)


def _write(out: BinaryIO, data: bytes) -> None:
    if out.write(data) != len(data):
        raise OSError("Short write while serializing GGUF")


def pad_to(
    out: BinaryIO, offset: int, chunk_size: int = DEFAULT_CHUNK, *, check: Check | None = None,
) -> None:
    remaining = offset - out.tell()
    if remaining < 0 or type(chunk_size) is not int or chunk_size <= 0:
        raise GGUFError("Invalid GGUF output padding range or chunk size")
    if remaining:
        zeros = bytes(min(remaining, chunk_size))
        while remaining:
            if check:
                check()
            size = min(remaining, len(zeros))
            _write(out, zeros if size == len(zeros) else zeros[:size])
            remaining -= size


def _write_string(out: BinaryIO, value: object) -> None:
    if not isinstance(value, str):
        raise GGUFError("GGUF STRING requires a text value")
    raw = value.encode("utf-8")
    if len(raw) > MAX_STRING_BYTES:
        raise GGUFError("GGUF string exceeds serialization limit")
    _write(out, struct.pack("<Q", len(raw)))
    _write(out, raw)


def _write_scalar(out: BinaryIO, type_id: int, value: object) -> None:
    if type_id == STRING:
        _write_string(out, value)
        return
    if type_id == BOOL and type(value) is not bool:
        raise GGUFError("GGUF BOOL requires a boolean value")
    if type_id in (0, 1, 2, 3, 4, 5, 10, 11) and type(value) is not int:
        raise GGUFError("GGUF integer metadata requires an integer value")
    if type_id in (6, 12) and type(value) not in (int, float):
        raise GGUFError("GGUF floating-point metadata requires a number")
    if type_id not in _SCALAR_FORMATS:
        raise GGUFError(f"Unsupported GGUF scalar type {type_id}")
    try:
        _write(out, struct.pack(_SCALAR_FORMATS[type_id], value))
    except (struct.error, OverflowError) as error:
        raise GGUFError(f"GGUF metadata value {value!r} does not fit {_VALUE_TYPES[type_id]}") from error


def _write_metadata(out: BinaryIO, key: str, type_id: int, value: object) -> None:
    _metadata_key(key)
    if type(type_id) is not int or not 0 <= type_id < len(_VALUE_TYPES):
        raise GGUFError(f"Unknown GGUF metadata type {type_id}")
    _write_string(out, key)
    _write(out, struct.pack("<I", type_id))
    if type_id == ARRAY:
        element_type, elements = value
        if (
            type(element_type) is not int or not 0 <= element_type < len(_VALUE_TYPES)
            or element_type == ARRAY or len(elements) > MAX_ARRAY_ELEMENTS
        ):
            raise GGUFError("Invalid GGUF array element type or count")
        _write(out, struct.pack("<IQ", element_type, len(elements)))
        for element in elements:
            _write_scalar(out, element_type, element)
    else:
        _write_scalar(out, type_id, value)


def _write_prefix(out: BinaryIO, version: int, tensor_count: int, metadata_count: int) -> None:
    if out.tell() != 0:
        raise GGUFError("GGUF header must start at file offset zero")
    if max(tensor_count, metadata_count) > MAX_RECORDS:
        raise GGUFError(f"GGUF count exceeds {MAX_RECORDS} tensor or metadata records")
    _write(out, _PREFIX.pack(b"GGUF", version, tensor_count, metadata_count))


def _write_tensor_header(
    out: BinaryIO, tensors: Sequence[Tensor], alignment: int, chunk_size: int,
    check: Check | None,
) -> int:
    expected = layout_tensors(
        ((tensor.name, tensor.shape, tensor.ggml_type) for tensor in tensors), alignment,
    )
    if tuple(tensors) != expected:
        raise GGUFError("GGUF output tensors have inconsistent offsets or sizes")
    for tensor in tensors:
        if check:
            check()
        _write_string(out, tensor.name)
        ndim = len(tensor.shape)
        _write(out, struct.pack(f"<I{ndim}QIQ", ndim, *tensor.shape, tensor.ggml_type, tensor.offset))
    data_offset = align_up(out.tell(), alignment) if tensors else out.tell()
    if data_offset > MAX_HEADER_BYTES:
        raise GGUFError("GGUF header exceeds serialization limit")
    pad_to(out, data_offset, chunk_size, check=check)
    return data_offset


def write_header(
    out: BinaryIO, source: GGUF, tensors: Sequence[Tensor],
    overrides: Mapping[str, object] | None = None, *, source_file: BinaryIO | None = None,
    chunk_size: int = DEFAULT_CHUNK, check: Check | None = None,
) -> int:
    """Write an aligned donor-derived header, copying unmodified metadata records verbatim."""
    overrides = {} if overrides is None else overrides
    for key in overrides:
        if key not in source.metadata_records:
            raise GGUFError(f"GGUF metadata override must address an existing key: {key!r}")
        if key in {"general.alignment", "general.quantization_version"} or source.metadata_records[key].type_id == ARRAY:
            raise GGUFError(f"GGUF metadata override is unsupported for {key!r}")
    _write_prefix(out, source.version, len(tensors), len(source.metadata_records))
    with source.open() if source_file is None else nullcontext(source_file) as stream:
        source.assert_unchanged(stream)
        for key, record in source.metadata_records.items():
            if key in overrides:
                _write_metadata(out, key, record.type_id, overrides[key])
            else:
                copy_range(stream, out, record.offset, record.nbytes, chunk_size, check=check)
        source.assert_unchanged(stream)
    return _write_tensor_header(out, tensors, source.alignment, chunk_size, check)


def write_fixture_header(
    out: BinaryIO, metadata: Mapping[str, tuple[int, object]], tensors: Sequence[Tensor],
) -> int:
    """Write a GGUF v3 header from explicitly typed fixture metadata."""
    alignment = DEFAULT_ALIGNMENT
    if "general.alignment" in metadata:
        type_id, value = metadata["general.alignment"]
        if type_id != UINT32:
            raise GGUFError("GGUF general.alignment must use UINT32")
        alignment = _validate_alignment(value)
    _write_prefix(out, 3, len(tensors), len(metadata))
    for key, (type_id, value) in metadata.items():
        _write_metadata(out, key, type_id, value)
    return _write_tensor_header(out, tensors, alignment, DEFAULT_CHUNK, None)
