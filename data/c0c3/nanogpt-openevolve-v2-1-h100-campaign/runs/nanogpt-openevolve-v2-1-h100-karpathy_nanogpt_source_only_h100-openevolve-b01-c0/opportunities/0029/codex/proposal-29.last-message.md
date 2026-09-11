MECHANISM: Redundant mid-stack full-attention removal

HYPOTHESIS: Using local attention in all patterned layers while retaining the forced final full-attention layer will beat 0.984313 val_bpb by increasing token throughput without eliminating full-sequence aggregation.

INTENDED_EDIT: Change `WINDOW_PATTERN` from `SSSL` to `SSSS`, reducing depth-8 full-attention layers from two to one; `_compute_window_sizes` still forces the final layer to full context.

EVIDENCE: The best run processed 506.2M tokens, while lower-throughput neighboring runs processed 500.2M and 500.7M and regressed; since max-autotune could not be verified, removing the intermediate full-attention layer is a directly verifiable throughput test.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # local attention throughout; final layer is forced to full context
>>>>>>> REPLACE