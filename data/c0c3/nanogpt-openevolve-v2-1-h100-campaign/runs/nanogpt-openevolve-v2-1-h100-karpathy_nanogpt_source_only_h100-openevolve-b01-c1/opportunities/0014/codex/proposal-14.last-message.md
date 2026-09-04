MECHANISM: Native unrestricted FlashAttention dispatch

HYPOTHESIS: Representing the essential final full-context layer with FA3’s unrestricted-window sentinel will preserve its receptive field while avoiding local-window boundary handling, increasing throughput enough to lower val_bpb below 0.992286.

INTENDED_EDIT: Replace the final layer’s explicit 2048-token local window with FA3’s semantically equivalent unrestricted attention mode.

EVIDENCE: Making the final layer local sharply regressed val_bpb to 1.015479, while the explicit full-context final layer achieved 0.992286; this targets implementation overhead without sacrificing the empirically essential global integration.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
=======
        window_sizes[-1] = (-1, -1)
>>>>>>> REPLACE