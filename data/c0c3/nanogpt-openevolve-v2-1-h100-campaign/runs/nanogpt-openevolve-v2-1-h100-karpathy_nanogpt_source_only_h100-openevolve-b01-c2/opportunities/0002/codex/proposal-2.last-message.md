MECHANISM: Single full-context anchor over stacked half-context attention

HYPOTHESIS: Replacing the intermediate full-context layer with half-context attention will process more than 497M tokens while preserving effective long-range propagation through stacked 1024-token windows, lowering val_bpb below 0.995558.

INTENDED_EDIT: Use half-context attention in the first seven layers while retaining the forced full-context final layer.

EVIDENCE: The current 1024-token short-window design achieved 0.995558 val_bpb on 497.0M tokens, whereas reducing windows to 512 unexpectedly cut throughput to 384.3M tokens and worsened val_bpb to 1.012775; this preserves the faster window size and removes only one full-attention layer.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # seven half-context layers; final layer is forced full-context
>>>>>>> REPLACE