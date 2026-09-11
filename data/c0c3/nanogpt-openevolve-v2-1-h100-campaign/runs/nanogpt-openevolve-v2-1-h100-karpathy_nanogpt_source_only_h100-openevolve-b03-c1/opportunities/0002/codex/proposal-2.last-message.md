MECHANISM: Eighth-context local attention with periodic global mixing

HYPOTHESIS: Reducing short-layer windows from 512 to 256 tokens will increase token throughput enough to lower val_bpb below 0.993365, while full-context layers 3 and 7 preserve long-range information.

INTENDED_EDIT: Use 256-token windows for “S” layers while retaining the existing SSSL pattern and full-context final layer.

EVIDENCE: Cutting short windows from 1024 to 512 improved val_bpb from 0.995558 to 0.993365 and increased training tokens from 497.0M to 512.2M, motivating the next controlled window-size ablation.

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