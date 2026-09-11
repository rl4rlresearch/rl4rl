MECHANISM: Half-context consolidation layer

HYPOTHESIS: A 1024-token final attention window will retain enough direct long-range integration while increasing throughput, lowering val_bpb below 0.992286.

INTENDED_EDIT: Keep seven 256-token local layers and reduce only the final layer’s window from 2048 to 1024 tokens.

EVIDENCE: A 256-token final layer regressed sharply to 1.015479, while a full-context final layer achieved 0.992286; testing the midpoint isolates how much final-layer context is necessary, using a previously performant 1024-token window size.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
=======
        window_sizes[-1] = (long_window // 2, 0)
>>>>>>> REPLACE