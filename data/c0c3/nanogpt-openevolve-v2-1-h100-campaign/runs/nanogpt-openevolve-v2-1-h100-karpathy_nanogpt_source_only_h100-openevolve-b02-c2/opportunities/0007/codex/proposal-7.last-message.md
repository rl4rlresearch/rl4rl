MECHANISM: Single global-attention consolidation layer

HYPOTHESIS: Replacing the layer-4 full-context attention with the proven 1024-token window will increase token throughput while the final full-context layer preserves global mixing, lowering val_bpb below 0.987182.

INTENDED_EDIT: Use 1024-token attention in the first seven layers and retain full-context attention only in the final layer.

EVIDENCE: The best 262K-batch design reached 0.987182 with 489.7M tokens. The 512-token experiment lost throughput, motivating reuse of the efficient 1024-token kernel while removing one redundant full-context layer.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # half context throughout; final layer is forced to full context
>>>>>>> REPLACE