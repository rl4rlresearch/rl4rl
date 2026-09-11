MECHANISM: Single global aggregation layer with efficient half-context attention

HYPOTHESIS: Using half-context attention in seven layers and retaining full context only in the final layer will beat val_bpb 0.995558 by processing more tokens while preserving final-layer global information flow.

INTENDED_EDIT: Change the window pattern from two full-context layers to one forced final full-context layer.

EVIDENCE: The baseline half-context design processed 497.0M tokens and reached 0.995558, outperforming all-full attention at 482.3M tokens and 0.999148; quarter-context was kernel-inefficient, so extending the proven half-context path is the most direct next test.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # half context throughout; final layer is forced to full context
>>>>>>> REPLACE