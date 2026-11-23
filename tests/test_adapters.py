from pathlib import Path
import struct
import tempfile
import unittest

from rustyspork.adapters import inspect_moe
from rustyspork.fixtures import create_fixture
from rustyspork.gguf import GGUFError, layout_tensors, open_gguf, pad_to, write_fixture_header


class MoeAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def _update_metadata_tail(self, path, key, trailing_offset, value):
        model = open_gguf(path)
        record = model.metadata_records[key]
        with path.open("r+b") as stream:
            stream.seek(record.offset + record.nbytes - trailing_offset)
            stream.write(struct.pack("<I", value))
        return open_gguf(path)

    def test_fixture_rejects_top_k_beyond_expert_count(self):
        path = create_fixture(self.base / "invalid-routing.gguf")
        model = self._update_metadata_tail(path, "lfm2moe.expert_used_count", 4, 5)
        with self.assertRaisesRegex(GGUFError, "top-k 5 exceeds"):
            inspect_moe(model)

    def test_fixture_marker_does_not_authorize_a_changed_graph(self):
        path = create_fixture(self.base / "invalid-graph.gguf")
        model = self._update_metadata_tail(path, "lfm2moe.attention.head_count_kv", 8, 4)
        with self.assertRaisesRegex(GGUFError, "fixture array lfm2moe.attention.head_count_kv"):
            inspect_moe(model)

    def _qwen(self, path, unexpected=()):
        metadata = {"general.architecture": (8, "qwen35moe")}
        metadata.update({f"qwen35moe.{name}": (4, value) for name, value in (
            ("expert_count", 4), ("expert_used_count", 2), ("block_count", 1),
            ("embedding_length", 8), ("expert_feed_forward_length", 4),
            ("expert_shared_feed_forward_length", 4),
        )})
        specs = (
            ("ffn_gate_inp.weight", (8, 4)),
            ("ffn_gate_exps.weight", (8, 4, 4)), ("ffn_up_exps.weight", (8, 4, 4)),
            ("ffn_down_exps.weight", (4, 8, 4)), ("ffn_gate_inp_shexp.weight", (8,)),
            ("ffn_gate_shexp.weight", (8, 4)), ("ffn_up_shexp.weight", (8, 4)),
            ("ffn_down_shexp.weight", (4, 8)),
        ) + unexpected
        tensors = layout_tensors([(f"blk.0.{name}", shape, 0) for name, shape in specs])
        with path.open("xb") as stream:
            data = write_fixture_header(stream, metadata, tensors)
            for tensor in tensors:
                pad_to(stream, data + tensor.offset)
                stream.write(bytes(tensor.nbytes))
        return open_gguf(path)

    def test_qwen_routed_identification_leaves_shared_experts_out(self):
        model = self._qwen(self.base / "qwen-layout.gguf")
        layout = inspect_moe(model)
        self.assertFalse(layout.can_compact)
        self.assertEqual({tensor.role: tensor.axis for tensor in layout.layers[0].tensors},
                         {"router": 1, "gate": 2, "up": 2, "down": 2})
        invalid = self._qwen(self.base / "qwen-extra-bias.gguf", (("ffn_gate_inp.bias", (4,)),))
        with self.assertRaisesRegex(GGUFError, "Unsupported.*ffn_gate_inp.bias"):
            inspect_moe(invalid)


if __name__ == "__main__":
    unittest.main()
