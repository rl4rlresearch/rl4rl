MECHANISM: Reduced local-attention span for higher token throughput

HYPOTHESIS: Reducing short-attention windows from 512 to 384 tokens will beat 0.984244 val_bpb by processing more training tokens while preserving the proven full-attention layers at depths 4, 6, and 8.

INTENDED_EDIT: Set every short-attention layer to three-sixteenths of the 2048-token sequence, preserving all other architecture, optimizer, batching, and schedule settings.

EVIDENCE: Expanding short windows to 768 reduced throughput from 499.6M to 488.6M tokens and regressed val_bpb from 0.984244 to 0.985340; testing a symmetric decrease probes whether the opposite context-throughput tradeoff improves the objective.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = 3 * long_window // 16
>>>>>>> REPLACE