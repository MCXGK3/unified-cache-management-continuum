# Turn-aware tiered storage backend

`UcmTurnTieredStore` is a storage adapter. It accepts caller-supplied core and
important block identities; it does not calculate attention scores or select KV.
The backend is registered with `UcmConnectorFactory` and can be used independently
of any particular sparse algorithm. Engine lifecycle integration is separate.

## Placement

- Full immutable backing is always created, dumped and committed in SSD via
  `UcmNfsStore`. A DRAM slice alone never produces a full-block `lookup()` hit.
- `publish()` accepts per-layer `core` and `important` hash lists, tensor shape
  and dtype. Only eligible important slices are promoted to pinned DRAM.
- HBM core views are supplied by the caller with a deadline. The caller must
  retain their physical allocations and call `release_core()` before reuse.
  This backend does not allocate or pin GPU KV slots by itself.
- `load()` resolves each slice in order: valid HBM view, DRAM slice, SSD range.
  GPU transfers use a separate stream; `wait()` confirms completion before use.

Call `begin_turn({"job_id": ..., "turn_id": ...})` as a conversation advances.
The DRAM window is configurable from one to three previous turns. Content keeps
its original per-conversation age even if retrieved or selected again. Capacity
is counted in actual K/V bytes; LRU eviction and age expiration leave SSD backing
intact. `cache.end_job(job_id)` releases that conversation's DRAM policy state.

## Configuration and interfaces

The standard NFS backend fields (`storage_backends`, `kv_block_size`, `io_size`,
`role`, `device`) are required. Additional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `dram_max_cache_size` | 128 MiB | Actual cached K/V slice byte budget; zero disables DRAM |
| `near_turns` | 3 | Historical turn window, from 1 to 3 |
| `representation_max_size` | 64 MiB | Separate bounded CPU representation metadata cache |

`load_host()` on `UcmNfsStore` reads file ranges directly into pinned CPU tensors.
Native NFS waits release the GIL, allowing other Python threads to submit GPU
work. Standard DRAM transfers also wait for the producing CUDA stream.

The `selections` argument to `publish()` is a dictionary keyed by integer layer:

```python
{0: {"core": [core_hash], "important": [important_hash],
     "shape": (block_size, num_kv_heads, head_size), "dtype": torch.float16}}
```

Offsets address K/V ranges inside a complete block file. Selection, full backup,
physical allocation ownership and caller-visible completion must be coordinated
by the engine adapter; choosing this backend alone does not install that adapter.
`save_manifest()` writes a caller-supplied manifest atomically. Restart recovery
of conversation state is not implemented by the backend.

## Verification

With the project environment active and a free CUDA GPU selected:

```bash
python test/test_turn_tiered_store.py
```

The four tests check caller-selected importance, turn expiration without age
refresh, byte capacity and shared content, plus real SSD/DRAM/HBM K/V recovery.
GPU-specific tests require the built `ucmnfsstore` native extension. Rebuild the
`ucmnfsstore` target after the C++ binding change in this commit.
