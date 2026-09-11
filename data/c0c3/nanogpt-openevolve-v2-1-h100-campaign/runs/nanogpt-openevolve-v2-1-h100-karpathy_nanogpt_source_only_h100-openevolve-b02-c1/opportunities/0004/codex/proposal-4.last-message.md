MECHANISM: Single global-context consolidation layer

HYPOTHESIS: Keeping half-context attention in the first seven layers and reserving full-context attention for the final layer will process more than 497M tokens while preserving global information flow, reducing val_bpb below 0.995558.

INTENDED_EDIT: Replace the SSSL pattern with half-context attention throughout; the existing window logic still forces the final layer to full context.

EVIDENCE: The all-full-context result processed only 417.9M tokens and regressed to 1.008973, while the mixed half-context baseline processed 497.0M tokens and reached 0.995558, indicating that reducing full-attention frequency is a promising throughput-quality tradeoff.

<<<<<<< SEARCH
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "S"    # half context throughout; final layer is forced full
>>>>>>> REPLACE