# All-Reduce

`all_reduce` is a two-rank GPU collective. It initializes one rank with `1`
and the other with `2`, sums them on the primary GPU, then broadcasts the
resulting `3` back to both ranks. `--verify` checks both GPU buffers.

Pantheon uses direct peer DMA when bidirectional peer access is available. If
the topology does not expose peer access, it reports a host-staged fallback
instead. The fallback validates correctness but should not be used as a direct
P2P bandwidth measurement.

```bash
pantheon --test all_reduce --duration 60 --gpu 0 --verify --profile
```

Use `--inject_error --verify` to confirm that verification detects a corrupted
value on the peer rank.
