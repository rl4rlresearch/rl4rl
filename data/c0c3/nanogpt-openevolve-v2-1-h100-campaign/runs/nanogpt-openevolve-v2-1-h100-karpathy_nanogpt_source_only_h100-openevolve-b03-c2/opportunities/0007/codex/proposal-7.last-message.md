MECHANISM: Single full-context anchor layer

HYPOTHESIS: Replacing the redundant mid-stack full-attention layer with half-context attention will preserve long-range modeling through the final full-context layer while increasing token throughput enough to beat the current 0.98713 val_bpb.

INTENDED_EDIT: Change the eight-layer attention pattern from six short and two full-context layers to seven short and one final full-context layer, retaining the verified 128-sequence, 262K-token update configuration.

EVIDENCE: The best design processed 489.7M tokens at 39.23% MFU, while lower-throughput designs processing 416.8M and 393.6M tokens regressed to 1.000243 and 0.997301 val_bpb; this motivates reducing attention compute without disturbing the efficient batch geometry.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # final layer is still forced to full context
>>>>>>> REPLACE