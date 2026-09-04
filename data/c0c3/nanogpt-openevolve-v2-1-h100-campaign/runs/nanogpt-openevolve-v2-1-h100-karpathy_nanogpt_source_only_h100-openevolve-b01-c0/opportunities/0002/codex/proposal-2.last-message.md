MECHANISM: Eighth-context local attention with periodic global mixing

HYPOTHESIS: Halving short-attention windows from 512 to 256 tokens will beat 0.993324 val_bpb by increasing token throughput, while full-attention layers 4 and 8 preserve long-range integration.

INTENDED_EDIT: Set S layers to one eighth of the 2048-token context while retaining the SSSL pattern and final full-attention layer.

EVIDENCE: Reducing short windows from 1024 to 512 increased training tokens from 497.0M to 514.3M and improved val_bpb from 0.995558 to 0.993324, motivating the next adjacent window-size ablation.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=eighth context
>>>>>>> REPLACE