"""L2 stores algorithm-selected shards; immutable full backing always uses L3.

This adapter never scores KV. Its inputs are (job, turn, layer, core hashes,
important hashes) emitted by the sparse method. DRAM capacity is measured in
actual bytes, including both K and V, rather than counting partial layers as
complete blocks. The scheduler's block pool accounts for L1 core references.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ucm.logger import init_logger
from ucm.store.nfsstore.nfsstore_connector import UcmNfsStore
from ucm.store.ucmstore import Task, UcmKVStoreBase

logger = init_logger(__name__)


@dataclass
class TurnTask(Task):
    ssd: object = None
    event: object = None
    references: list = field(default_factory=list)
    promote: list = field(default_factory=list)


class TurnCache:
    """CPU policy state, separately testable without a GPU or storage backend."""

    def __init__(self, capacity, near_turns=3):
        if capacity < 0 or not 1 <= near_turns <= 3:
            raise ValueError("DRAM capacity must be >= 0 and near_turns in [1, 3]")
        self.capacity = capacity
        self.near_turns = near_turns
        self.latest = {}
        self.owners = {}  # immutable content hash -> set of original (job, turn)
        self.selections = {}  # (job, turn, layer) -> important hash set
        self.shards = OrderedDict()  # (hash, byte offset) -> pinned CPU tensor
        self.bytes = 0
        self.lock = threading.RLock()

    def begin(self, job, turn):
        with self.lock:
            self.latest[job] = turn
            # Newer selections do not refresh the age of old KV. Age follows
            # the turn that produced the block, not the last retrieval time.
            for identity in list(self.selections):
                if identity[0] == job and turn - identity[1] > self.near_turns:
                    del self.selections[identity]
            self.trim()

    def register(self, job, turn, hashes):
        with self.lock:
            for key in hashes:
                owners = self.owners.setdefault(key, set())
                # A cumulative prompt contains old blocks; don't reassign them.
                if not any(owner[0] == job for owner in owners):
                    owners.add((job, turn))

    def select(self, job, turn, layer, important):
        with self.lock:
            self.selections[job, turn, layer] = set(important)
            self.trim()

    def eligible(self, key, layer):
        return any(
            0 <= self.latest.get(job, turn) - turn <= self.near_turns
            and any(key in hashes for (j, _t, l), hashes in self.selections.items()
                    if j == job and l == layer)
            for job, turn in self.owners.get(key, ()))

    def trim(self):
        for shard in list(self.shards):
            key, _offset, layer = shard
            if not self.eligible(key, layer):
                self._remove(shard)

    def _remove(self, shard):
        tensor = self.shards.pop(shard)
        self.bytes -= tensor.numel() * tensor.element_size()

    def put(self, key, offset, layer, tensor):
        with self.lock:
            size = tensor.numel() * tensor.element_size()
            if not self.eligible(key, layer) or size > self.capacity:
                return False
            shard = key, offset, layer
            if shard in self.shards:
                self._remove(shard)
            while self.bytes + size > self.capacity:
                self._remove(next(iter(self.shards)))
            self.shards[shard] = tensor
            self.bytes += size
            return True

    def get(self, key, offset, layer):
        with self.lock:
            shard = key, offset, layer
            value = self.shards.get(shard)
            if value is not None:
                self.shards.move_to_end(shard)
            return value

    def end_job(self, job):
        with self.lock:
            self.latest.pop(job, None)
            for identity in list(self.selections):
                if identity[0] == job:
                    del self.selections[identity]
            for key in list(self.owners):
                self.owners[key] = {owner for owner in self.owners[key] if owner[0] != job}
                if not self.owners[key]:
                    del self.owners[key]
            self.trim()


class UcmTurnTieredStore(UcmKVStoreBase):
    def __init__(self, config):
        super().__init__(config)
        self.ssd = UcmNfsStore(dict(config))
        self.cache = TurnCache(int(config.get("dram_max_cache_size", 128 * 1024**2)),
                               int(config.get("near_turns", 3)))
        self.layer_bytes = int(config["io_size"]) * 2
        self.l1 = {}  # request -> (deadline, job, {(hash, offset): GPU view})
        self.manifest_dir = Path(config["storage_backends"].split(":")[0]) / "turn_manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hbm_hits": 0, "dram_hits": 0, "ssd_loads": 0,
                      "promotions": 0, "backed_blocks": 0, "feature_hits": 0}
        self.features = OrderedDict()
        self.feature_bytes = 0
        self.feature_capacity = int(config.get("representation_max_size", 64 * 1024**2))

    def get_features(self, layer, hashes):
        values = []
        for key in hashes:
            value = self.features.get((layer, key))
            if value is not None:
                self.features.move_to_end((layer, key))
                self.stats["feature_hits"] += 1
            values.append(value)
        return values

    def remember_features(self, layer, hashes, features):
        # Bounded algorithm metadata, independent of the KV byte budget. No
        # scoring or importance filtering is performed by this cache.
        for key, value in zip(hashes, features):
            identity = layer, key
            if identity in self.features:
                continue
            size = value.numel() * value.element_size()
            if size > self.feature_capacity:
                continue
            while self.feature_bytes + size > self.feature_capacity:
                _, old = self.features.popitem(last=False)
                self.feature_bytes -= old.numel() * old.element_size()
            self.features[identity] = value.detach().cpu().clone()
            self.feature_bytes += size

    def cc_store(self):
        return 0

    def create(self, ids):
        return self.ssd.create(ids)

    def lookup(self, ids):
        # A DRAM shard is not evidence that the entire block has a backup.
        return self.ssd.lookup(ids)

    def dump(self, ids, offsets, tensors):
        return self.ssd.dump(ids, offsets, tensors)

    def commit(self, ids, is_success=True):
        self.ssd.commit(ids, is_success)
        if is_success:
            self.stats["backed_blocks"] += len(ids)

    def begin_turn(self, meta):
        self.cache.begin(meta["job_id"], meta["turn_id"])

    def release_core(self, request_id):
        self.l1.pop(request_id, None)

    def _hbm(self, key, offset):
        now = time.time()
        for rid, (deadline, _job, tensors) in list(self.l1.items()):
            if deadline <= now:
                self.l1.pop(rid, None)
            elif (key, offset) in tensors:
                return tensors[key, offset]
        return None

    def load(self, ids, offsets, tensors):
        task = TurnTask()
        misses, miss_offsets, miss_tensors = [], [], []
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for key, offset, tensor in zip(ids, offsets, tensors):
                layer = offset // self.layer_bytes
                src = self._hbm(key, offset)
                tier = "hbm_hits"
                if src is None:
                    src = self.cache.get(key, offset, layer)
                    tier = "dram_hits"
                if src is not None:
                    tensor.copy_(src, non_blocking=True)
                    task.references.append(src)  # keep evicted CPU storage alive
                    self.stats[tier] += 1
                else:
                    misses.append(key)
                    miss_offsets.append(offset)
                    miss_tensors.append(tensor)
                    self.stats["ssd_loads"] += 1
                    if self.cache.eligible(key, layer):
                        task.promote.append((key, offset, layer, tensor))
            task.event = torch.cuda.Event()
            task.event.record(stream)
        if misses:
            task.ssd = self.ssd.load(misses, miss_offsets, miss_tensors)
        return task

    def wait(self, task):
        if not isinstance(task, TurnTask):
            return self.ssd.wait(task)
        if task.ssd is not None and self.ssd.wait(task.ssd) != 0:
            return -1
        task.event.synchronize()
        # Called by ESA's transfer thread. Staging/promotion does not block the
        # model's Python thread and only starts after the native load completed.
        for key, offset, layer, tensor in task.promote:
            host = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            if self.cache.put(key, offset, layer, host):
                self.stats["promotions"] += 1
        return 0

    def publish(self, meta, hashes, selections, core_views, deadline):
        job, turn = meta["job_id"], meta["turn_id"]
        self.cache.register(job, turn, hashes)
        for layer, result in selections.items():
            self.cache.select(job, turn, layer, result["important"])
            if self.cache.capacity == 0:
                continue
            # Materialize only algorithm-selected slices, from the immutable
            # backup. This also works when their HBM slots were already reused.
            for key in result["important"]:
                if not self.cache.eligible(key, layer):
                    continue
                for is_v in (0, 1):
                    offset = layer * self.layer_bytes + is_v * self.layer_bytes // 2
                    if self.cache.get(key, offset, layer) is not None:
                        continue
                    # NFS host API uses bytes, so dtype/layout are restored by
                    # the destination tensor's copy in load().
                    shape, dtype = result["shape"], result["dtype"]
                    host = torch.empty(shape, dtype=dtype, pin_memory=True)
                    task = self.ssd.load_host([key], [offset], [host])
                    if self.ssd.wait(task) != 0:
                        raise RuntimeError("Turn-tiered important-shard promotion failed")
                    if self.cache.put(key, offset, layer, host):
                        self.stats["promotions"] += 1
        if deadline > time.time() and core_views:
            self.l1[meta["request_id"]] = deadline, job, core_views
        logger.info("UCM_TURN_FINISH job=%s turn=%d dram_bytes=%d stats=%s",
                    job, turn, self.cache.bytes, self.stats)

    def save_manifest(self, name, manifest):
        # Content-address the filename; never interpolate user job IDs as paths.
        import hashlib
        path = self.manifest_dir / (hashlib.sha256(name.encode()).hexdigest() + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False))
        os.replace(temporary, path)

    def prefetch(self, ids):
        pass

    def fetch_data(self, *args):
        raise NotImplementedError("Turn-aware loads require tensor layout metadata")

    def dump_data(self, *args):
        raise NotImplementedError("Turn-aware dumps require tensor layout metadata")

    def check(self, task):
        if isinstance(task, TurnTask):
            return 0, task.event.query() and not task.promote and task.ssd is None
        return self.ssd.check(task)
