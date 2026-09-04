MECHANISM: Intermediate local-attention compute allocation

HYPOTHESIS: Using 384-token short-layer attention will reduce `val_bpb` below 0.984156 by preserving more local context than the regressing 256-token design while exposing more training tokens than the best 512-token design.

INTENDED_EDIT: Set the six short attention windows to 3/16 of the 2048-token sequence length, retaining two full-context layers and all verified optimizer schedules.

EVIDENCE: Reducing short attention from 1024 to 512 improved `val_bpb` from 0.985708 to 0.984156, while a further reduction to 256 slightly regressed to 0.984221 despite higher throughput; this brackets a likely context-throughput optimum between 256 and 512 tokens.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 3 * long_window // 16
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=3/16 context
>>>>>>> REPLACE