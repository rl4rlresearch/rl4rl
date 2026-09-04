MECHANISM: Three-quarter-context consolidation layer

HYPOTHESIS: A 1536-token final attention window will preserve more long-range integration than the slightly worse 1024-token variant while reducing attention work versus 2048 tokens, lowering val_bpb below 0.992286.

INTENDED_EDIT: Keep seven 256-token local layers and change only the final layer’s window from 2048 to 1536 tokens.

EVIDENCE: Full-context final attention achieved 0.992286, while 1024-token final attention remained close at 0.992829; this motivates a finer context-throughput test between them.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
=======
        window_sizes[-1] = (3 * long_window // 4, 0)
>>>>>>> REPLACE