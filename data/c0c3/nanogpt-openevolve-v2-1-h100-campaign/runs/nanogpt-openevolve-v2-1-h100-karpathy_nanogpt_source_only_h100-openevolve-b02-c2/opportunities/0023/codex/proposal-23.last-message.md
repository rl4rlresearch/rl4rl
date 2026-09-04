MECHANISM: Deeper local-window compression with preserved global integration

HYPOTHESIS: Reducing the six local-attention layers from 768 to 512 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.984868 by increasing token throughput without removing the proven two-layer global-context path.

INTENDED_EDIT: Restore the best two-global-layer SSSL pattern and halve local attention windows to one-quarter context.

EVIDENCE: Reducing local windows from 1024 to 768 while preserving full-context layers 4 and 8 improved val_bpb from 0.985746 to 0.984868 and raised processed tokens from 491.3M to 496.0M; this motivates testing the next 256-token reduction.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # use half context except for the always-full final layer
=======
WINDOW_PATTERN = "SSSL" # 512-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE