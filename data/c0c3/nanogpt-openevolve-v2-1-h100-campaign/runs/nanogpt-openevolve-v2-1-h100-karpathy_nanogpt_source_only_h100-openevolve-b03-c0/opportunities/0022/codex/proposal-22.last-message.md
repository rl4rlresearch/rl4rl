MECHANISM: Finer-grained multi-head attention

HYPOTHESIS: Doubling full-capacity query/KV heads from 4 to 8 while preserving the 512-dimensional residual stream and parameter count will achieve `val_bpb < 0.985780` without materially reducing throughput.

INTENDED_EDIT: Reduce `HEAD_DIM` from 128 to 64, yielding eight full KV attention heads instead of four while leaving depth, model width, batching, optimizer, schedule, and softcap unchanged.

EVIDENCE: Halving KV heads improved throughput but regressed `val_bpb` from `0.985780` to `0.992933`, showing attention-head structure materially affects quality; this tests the opposite direction without reducing total KV dimension or model capacity.

<<<<<<< SEARCH
HEAD_DIM = 128          # target head dimension for attention
=======
HEAD_DIM = 64           # target head dimension for attention
>>>>>>> REPLACE