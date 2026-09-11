MECHANISM: Finer-grained multi-head attention

HYPOTHESIS: Reducing head dimension from 128 to 64 will beat 0.984313 val_bpb by doubling attention heads from 4 to 8 while preserving model width, matrix sizes, and nominal attention FLOPs.

INTENDED_EDIT: Set `HEAD_DIM` to 64, retaining the proven architecture, optimizer, batch size, and 78.0% linear warmdown.

EVIDENCE: The adjacent 77.9% and 78.1% warmdown tests both regressed from the 78.0% result, motivating an orthogonal attention-granularity test that leaves the established compute scale essentially unchanged.

<<<<<<< SEARCH
HEAD_DIM = 128          # target head dimension for attention
=======
HEAD_DIM = 64           # target head dimension for attention
>>>>>>> REPLACE