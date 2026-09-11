MECHANISM: Depth-preserving local-attention contraction

HYPOTHESIS: Quarter-context short-window layers will process more than 486.2M tokens while preserving two full-context layers, achieving `val_bpb < 0.986636`.

INTENDED_EDIT: Reduce short attention windows from 1024 to 512 tokens without changing depth, model width, batching, or optimizer settings.

EVIDENCE: Removing a transformer block increased throughput to 539.9M tokens but worsened `val_bpb` to 0.990593, while contracting the MLP also regressed; shortening attention instead preserves all eight blocks and the efficient 2048-wide MLP while reducing sequence-dependent compute.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE