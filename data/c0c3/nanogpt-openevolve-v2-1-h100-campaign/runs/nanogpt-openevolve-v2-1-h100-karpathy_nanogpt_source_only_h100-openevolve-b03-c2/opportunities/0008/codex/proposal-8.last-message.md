MECHANISM: Two-way grouped-query attention with efficient single-microbatch updates

HYPOTHESIS: Using four query heads but only two shared KV heads on the proven 128-sequence, 262K-token update geometry will beat val_bpb 0.98713 by reducing projection, value-embedding, and optimizer overhead while retaining query capacity and full token throughput.

INTENDED_EDIT: Restore the best verified single-microbatch configuration and replace full multi-head KV attention with two-way grouped-query attention.

EVIDENCE: Reference Design 3 achieved the best val_bpb, 0.98713, at 489.7M tokens and 39.23% MFU; lower-throughput variants regressed, motivating an architecture reduction that preserves its efficient batch geometry.

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18     # ~262K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # two microbatches per optimizer step
=======
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE