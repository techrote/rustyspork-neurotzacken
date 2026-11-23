import hashlib
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path

from rustyspork.gguf import (
    GGUFError,
    align_up,
    copy_range,
    layout_tensors,
    open_gguf,
    pad_to,
    write_fixture_header,
    write_header,
)


class GGUFTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.folder = Path(workspace.name)
        self.path = self.folder / "donor.gguf"
        self.tensors = layout_tensors((("expert.q6", (256, 2, 4), 14), ("router", (3, 4), 0)))
        self.metadata = {
            "general.architecture": (8, "example"),
            "general.quantization_version": (4, 2),
            "example.expert_count": (4, 4),
            "tokenizer.ggml.tokens": (9, (8, ("[unused]", "λ", "with\x00nul"))),
        }
        with self.path.open("wb") as out:
            self.data_offset = write_fixture_header(out, self.metadata, self.tensors)
            for index, tensor in enumerate(self.tensors):
                pad_to(out, self.data_offset + tensor.offset)
                out.write(bytes([index + 13]) * tensor.nbytes)
            pad_to(out, align_up(out.tell(), 32))

    def test_inventory_and_exact_header_round_trip(self):
        source = open_gguf(self.path)
        inventory = source.inventory(chunk_size=73)
        expert = inventory["tensors"][0]
        self.assertEqual(expert["logical_elements"], 2048)
        self.assertEqual(expert["storage_bytes"], 1680)
        self.assertEqual(expert["file_offset_bytes"], self.data_offset)
        self.assertEqual(inventory["source_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(source.metadata_array("tokenizer.ggml.tokens"), ("[unused]", "λ", "with\x00nul"))
        with self.assertRaisesRegex(GGUFError, "element limit"):
            source.metadata_array("tokenizer.ggml.tokens", max_elements=2)
        copied = io.BytesIO()
        with source.open() as donor:
            self.assertEqual(write_header(copied, source, self.tensors, source_file=donor), self.data_offset)
            copy_range(donor, copied, source.data_offset, source.size - source.data_offset, chunk_size=71)
        self.assertEqual(copied.getvalue(), self.path.read_bytes())

    def test_unsupported_version_and_malformed_headers(self):
        raw = self.path.read_bytes()
        table_start = raw.index(struct.pack("<Q", len("expert.q6")) + b"expert.q6")
        shape_start = table_start + 8 + len("expert.q6") + 4
        type_start = shape_start + 3 * 8
        mutations = [
            (4, struct.pack(">I", 3), "Big-endian"),
            (4, struct.pack("<I", 1), "Unsupported GGUF version"),
            (4, struct.pack("<I", 4), "Unsupported GGUF version"),
            (shape_start, struct.pack("<Q", 255), "block width"),
            (type_start, struct.pack("<I", 4), "Unsupported GGML type"),
            (type_start + 4, struct.pack("<Q", 32), "expected packed aligned offset"),
            (len(raw) - 64, b"", "payload extends beyond"),
        ]
        malformed = self.folder / "malformed.gguf"
        for offset, replacement, message in mutations:
            with self.subTest(message=message):
                suffix = raw[offset + len(replacement):] if replacement else b""
                malformed.write_bytes(raw[:offset] + replacement + suffix)
                with self.assertRaisesRegex(GGUFError, message):
                    open_gguf(malformed)

    def test_supported_v2_and_quantization_version(self):
        raw = self.path.read_bytes()
        path = self.folder / "versioned.gguf"
        path.write_bytes(raw[:4] + struct.pack("<I", 2) + raw[8:])
        self.assertEqual(open_gguf(path).version, 2)
        key = b"general.quantization_version"
        key_offset = raw.index(key)
        type_offset = key_offset + len(key)
        path.write_bytes(raw[:type_offset] + struct.pack("<II", 4, 1) + raw[type_offset + 8:])
        with self.assertRaisesRegex(GGUFError, "Unsupported GGML quantization version"):
            open_gguf(path)
        path.write_bytes(raw[:type_offset] + struct.pack("<I", 5) + raw[type_offset + 4:])
        with self.assertRaisesRegex(GGUFError, "must use UINT32"):
            open_gguf(path)
        path.write_bytes(raw[:key_offset] + b"x" + raw[key_offset + 1:])
        with self.assertRaisesRegex(GGUFError, "Quantized GGUF requires"):
            open_gguf(path)

    def test_duplicate_keys_and_invalid_array_element_type(self):
        key = b"general.name"
        record = struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 1)
        bad = self.folder / "bad-metadata.gguf"
        bad.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 2) + record * 2)
        with self.assertRaisesRegex(GGUFError, "Duplicate GGUF metadata key"):
            open_gguf(bad)
        record_prefix = struct.pack("<Q", len(key)) + key + struct.pack("<I", 9)
        bad.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) + record_prefix + struct.pack("<IQ", 9, 0))
        with self.assertRaisesRegex(GGUFError, "Nested GGUF arrays"):
            open_gguf(bad)

    def test_source_snapshot_and_hash_checks(self):
        source = open_gguf(self.path)
        checks = []
        source.sha256(chunk_size=113, check=lambda: checks.append(None))
        self.assertGreater(len(checks), 2)
        replacement = self.folder / "replacement.gguf"
        replacement.write_bytes(self.path.read_bytes())
        os.replace(replacement, self.path)
        with self.assertRaisesRegex(GGUFError, "source changed"):
            source.sha256()
