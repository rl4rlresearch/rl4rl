MECHANISM: Single terminal global-attention layer

HYPOTHESIS: Retaining 512-token local windows but reducing full-attention layers from two to one will beat 0.993324 val_bpb by increasing throughput while seven stacked local layers and the final global layer preserve full-context access.

INTENDED_EDIT: Change the attention pattern so layers 1–7 use 512-token windows and only layer 8 uses full attention.

EVIDENCE: The 512-token design improved val_bpb to 0.993324 with 514.3M tokens, while shrinking windows to 256 reduced throughput and regressed to 0.998036; this tests a different way to remove attention compute without leaving the demonstrated 512-token operating point.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "S"    # 512-token local attention; final layer is forced full
>>>>>>> REPLACE