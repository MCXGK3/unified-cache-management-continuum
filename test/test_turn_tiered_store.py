"""Storage policy and real SSD/DRAM/HBM transfers, independent of ESA."""

import hashlib
import tempfile
import time
import unittest

import torch

from ucm.store.tieredstore.turn_tiered_store import TurnCache, UcmTurnTieredStore


class TurnStoreTests(unittest.TestCase):
    def test_only_algorithm_important_data_enters_dram(self):
        cache = TurnCache(128)
        cache.begin("j", 0)
        cache.register("j", 0, ["core", "important", "other"])
        cache.select("j", 0, 0, ["important"])
        tensor = torch.zeros(8)
        self.assertFalse(cache.put("core", 0, 0, tensor))
        self.assertFalse(cache.put("other", 0, 0, tensor))
        self.assertTrue(cache.put("important", 0, 0, tensor))
        self.assertIsNone(cache.get("important", 0, 1))

    def test_remote_history_does_not_become_near_on_retrieval(self):
        cache = TurnCache(128, near_turns=3)
        cache.begin("j", 0)
        cache.register("j", 0, ["old"])
        cache.select("j", 0, 0, ["old"])
        cache.put("old", 0, 0, torch.zeros(8))
        cache.begin("j", 3)
        self.assertIsNotNone(cache.get("old", 0, 0))
        cache.begin("j", 4)
        cache.register("j", 4, ["old"])
        cache.select("j", 4, 0, ["old"])
        self.assertFalse(cache.eligible("old", 0))
        self.assertEqual(cache.bytes, 0)

    def test_shared_content_and_actual_byte_budget(self):
        cache = TurnCache(32)
        for job in ("a", "b"):
            cache.begin(job, 0)
            cache.register(job, 0, ["shared", "second"])
            cache.select(job, 0, 0, ["shared", "second"])
        cache.put("shared", 0, 0, torch.zeros(8))
        cache.put("second", 0, 0, torch.zeros(8))
        self.assertEqual(cache.bytes, 32)
        self.assertIsNone(cache.get("shared", 0, 0))
        cache.end_job("a")
        self.assertIsNotNone(cache.get("second", 0, 0))
        cache.end_job("b")
        self.assertEqual(cache.bytes, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_real_ssd_backup_dram_promotion_and_expiration(self):
        with tempfile.TemporaryDirectory(prefix="turn-store-test-") as directory:
            store = UcmTurnTieredStore({"role": "worker", "device": 0,
                "kv_block_size": 1024, "io_size": 256,
                "dram_max_cache_size": 2048, "near_turns": 3,
                "storage_backends": directory, "use_direct": False,
                "stream_number": 2, "buffer_number": 4})
            # Native NFS block identities are 128-bit hashes, not arbitrary
            # short user strings (matching RequestHasher's production output).
            keys = [str(int(hashlib.md5(str(i).encode()).hexdigest(), 16)) for i in range(3)]
            self.assertEqual(store.create(keys), [0, 0, 0])
            tensors = [torch.full((16, 2, 4), i + 1., device="cuda", dtype=torch.float16)
                       for i in range(12)]
            names = keys * 4
            offsets = [offset for offset in (0, 256, 512, 768) for _ in keys]
            torch.cuda.synchronize()
            self.assertEqual(store.wait(store.dump(names, offsets, tensors)), 0)
            store.commit(keys)
            meta = {"job_id": "job", "turn_id": 0, "request_id": "r"}
            store.begin_turn(meta)
            selections = {layer: {"core": [keys[0]], "important": [keys[1]],
                                  "shape": (16, 2, 4), "dtype": torch.float16}
                          for layer in (0, 1)}
            core = {(keys[0], 0): tensors[0]}
            store.publish(meta, keys, selections, core, time.time() + 10)
            self.assertEqual(store.cache.bytes, 1024)
            self.assertEqual(store.lookup(keys), [True, True, True])
            restored = [torch.zeros_like(tensor) for tensor in tensors]
            self.assertEqual(store.wait(store.load(names, offsets, restored)), 0)
            for left, right in zip(tensors, restored):
                torch.testing.assert_close(left, right)
            self.assertGreater(store.stats["hbm_hits"], 0)
            self.assertGreater(store.stats["dram_hits"], 0)
            store.release_core("r")
            store.begin_turn({"job_id": "job", "turn_id": 4})
            self.assertEqual(store.cache.bytes, 0)
            before = store.stats["ssd_loads"]
            self.assertEqual(store.wait(store.load(names, offsets, restored)), 0)
            self.assertEqual(store.stats["ssd_loads"] - before, 12)
            for left, right in zip(tensors, restored):
                torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
