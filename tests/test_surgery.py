import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rustyspork.adapters import inspect_moe
from rustyspork.fixtures import create_fixture
from rustyspork.gguf import GGUFError, Tensor, open_gguf
from rustyspork.lab import compact, load_json
from rustyspork.surgery import Budget, expert_edit, make_plan, prepare, q6_value, verify_candidate


class SurgeryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = self.root / "runs"

    def source(self, count=4):
        return open_gguf(create_fixture(self.root / f"moe-{count}.gguf", count))

    def plan(self, model, keep):
        return make_plan(model, Budget(), operation="gouge", keep=keep)

    def candidate(self, receipt):
        return open_gguf(self.runs / receipt["run_id"] / "candidate.gguf")

    @staticmethod
    def tensor_bytes(model, name):
        tensor = model.tensors[name]
        with model.open() as stream:
            stream.seek(model.data_offset + tensor.offset)
            return stream.read(tensor.nbytes)

    def test_structural_repack_by_layer_preserves_exact_blocks(self):
        for count, keeps, saved in (
            (4, [[0, 2], [1, 3]], 649216), (8, [[0, 2, 4, 6], [1, 3, 5, 7]], 1298432),
        ):
            with self.subTest(experts=count):
                donor = self.source(count)
                source_hash = donor.sha256()
                plan = self.plan(donor, keeps[0])
                plan["operation"]["layers"][1]["survivors"] = keeps[1]
                preview = prepare(donor, plan, Budget()).summary()
                self.assertEqual(preview["predicted_file_bytes_saved"], saved)
                receipt = compact(donor, plan, self.runs, Budget())
                self.assertEqual(receipt, compact(donor, plan, self.runs, Budget()))
                candidate = self.candidate(receipt)
                self.assertEqual(inspect_moe(candidate).expert_count, count // 2)
                self.assertEqual(candidate.size, donor.size - saved)
                self.assertEqual(donor.sha256(), source_hash)
                self.assertEqual(receipt["verification"], verify_candidate(prepare(donor, plan, Budget()), candidate, Budget()))
                for layer, keep in enumerate(keeps):
                    for name, stride in (("ffn_gate_exps.weight", 256 * 210),
                                         ("ffn_gate_inp.weight", 256 * 4), ("exp_probs_b.bias", 4)):
                        full_name = f"blk.{layer}.{name}"
                        original = self.tensor_bytes(donor, full_name)
                        expected = b"".join(original[i * stride:(i + 1) * stride] for i in keep)
                        self.assertEqual(self.tensor_bytes(candidate, full_name), expected)
                self.assertEqual(self.tensor_bytes(candidate, "token_embd.weight"), self.tensor_bytes(donor, "token_embd.weight"))
                self.assertEqual(receipt["verification"]["expert_mappings"][1]["old_to_new"][keeps[1][-1]], count // 2 - 1)

    def test_zero_removal_and_corruption(self):
        donor = self.source()
        plan = self.plan(donor, [0, 1, 2, 3])
        receipt = compact(donor, plan, self.runs, Budget())
        candidate = self.candidate(receipt)
        self.assertEqual(donor.path.read_bytes(), candidate.path.read_bytes())
        self.assertEqual(receipt["verification"]["candidate_sha256"], donor.sha256())
        with candidate.path.open("r+b") as stream:
            stream.seek(candidate.data_offset + candidate.tensors["token_embd.weight"].offset)
            stream.write(b"\x01")
        with self.assertRaisesRegex(GGUFError, "bytes differ"):
            verify_candidate(prepare(donor, plan, Budget()), open_gguf(candidate.path), Budget())
        with self.assertRaisesRegex(GGUFError, "bytes differ"):
            compact(donor, plan, self.runs, Budget())

    def test_invalid_plans_and_budget(self):
        donor = self.source()
        plan = self.plan(donor, [0, 2])
        for ids, message in (([0, 0], "unique"), ([0, 4], "invalid index"), ([0], "top-2")):
            with self.subTest(ids=ids):
                modified = copy.deepcopy(plan)
                modified["operation"]["layers"][0]["survivors"] = ids
                with self.assertRaisesRegex(GGUFError, message):
                    prepare(donor, modified, Budget())
        wrong = {**plan, "source_sha256": "0" * 64}
        with self.assertRaisesRegex(GGUFError, "SHA-256"):
            prepare(donor, wrong, Budget())
        with self.assertRaisesRegex(GGUFError, "output budget"):
            compact(donor, plan, self.runs, Budget(max_output_bytes=100))
        self.assertEqual(len(list(self.runs.glob("*.failure-*.json"))), 1)
        self.assertFalse(list(self.runs.glob("*/candidate.gguf")))
        with self.assertRaisesRegex(GGUFError, "split Q6_K storage blocks"):
            expert_edit(Tensor("quantized", (256, 4), 14, 0, 840), 0, (0, 1))

    def test_invalid_router_and_interruption_leave_no_published_model(self):
        donor = self.source()
        original_hash = donor.sha256()
        plan = self.plan(donor, [0, 2])
        with patch("rustyspork.lab.write_candidate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                compact(donor, plan, self.runs, Budget())
        self.assertEqual(list(self.runs.glob("*.partial")), [])
        self.assertEqual(list(self.runs.glob("*/candidate.gguf")), [])
        audit, = self.runs.glob("*.failure-*.json")
        self.assertEqual(load_json(audit, 1024 * 1024)["state"], "cancelled")
        self.assertEqual(original_hash, donor.sha256())
        compact(donor, plan, self.runs, Budget())

        invalid_path = self.root / "invalid-router.gguf"
        invalid_path.write_bytes(donor.path.read_bytes())
        with invalid_path.open("r+b") as stream:
            stream.seek(donor.data_offset + donor.tensors["blk.0.exp_probs_b.bias"].offset)
            stream.write(struct.pack("<f", float("nan")))
        with self.assertRaisesRegex(GGUFError, "router or bias contains a non-finite value"):
            self.plan(open_gguf(invalid_path), [0, 2])

    def test_scoop_scrape_and_probe(self):
        donor = self.source()
        kwargs = {"tensor_name": "blk.0.exp_probs_b.bias", "selected": [1, 3]}
        scoop = make_plan(donor, Budget(), operation="scoop", scale=0, **kwargs)
        receipt = compact(donor, scoop, self.runs, Budget())
        output = self.candidate(receipt)
        self.assertEqual(struct.unpack("<4f", self.tensor_bytes(output, kwargs["tensor_name"])), (0, 0, 0.25, 0))
        self.assertEqual([row["index"] for row in receipt["verification"]["numeric_observations"] if row["bytes_changed"]], [1, 3])
        scrape = make_plan(donor, Budget(), operation="scrape", amplitude=0.5, seed=42, **kwargs)
        first = compact(donor, scrape, self.runs, Budget())["verification"]
        second = compact(donor, scrape, self.root / "repeat", Budget())["verification"]
        self.assertEqual(first, second)
        self.assertNotEqual(first["source_sha256"], first["candidate_sha256"])
        probe = make_plan(donor, Budget(), operation="probe", tensor_name="blk.0.ffn_gate_exps.weight",
                          selected=[0, 255, 256, 65536])
        values = prepare(donor, probe, Budget()).probe
        self.assertEqual([row["value"] for row in values["samples"]], [1 / 16384, 1 / 16384, 49 / 16384, 2 / 16384])
        with self.assertRaisesRegex(GGUFError, "exact source inventory"):
            probe["operation"]["tensor"]["shape"][0] += 1
            prepare(donor, probe, Budget())
        self.assertEqual(hashlib.sha256(donor.path.read_bytes()).hexdigest(), donor.sha256())

    def test_q6_probe_uses_lane_aware_nibbles_and_signed_scales(self):
        # The four groups share packed nibbles and upper bits in each half.
        block = bytes([0x31]) * 32 + bytes([0x42]) * 32 + bytes([0x75]) * 32 + bytes([0x86]) * 32
        block += bytes([0xE4]) * 32 + bytes([0x1B]) * 32
        block += struct.pack("<16b", *range(-8, 8)) + struct.pack("<e", 0.5)
        self.assertEqual([q6_value(block, i) for i in (0, 16, 32, 64, 96, 128, 160, 192, 224)],
                         [124, 108.5, 42, -6, -20, 0, 6, -18, -72])


if __name__ == "__main__":
    unittest.main()
