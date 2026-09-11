MECHANISM: Single global-attention consolidation layer

HYPOTHESIS: Replacing the earlier full-context layer with a 256-token local layer will increase token throughput while the mandatory final full-context layer preserves global integration, lowering val_bpb below 0.993215.

INTENDED_EDIT: Use local attention in every layer except the final layer, which remains full-context through the existing override.

EVIDENCE: Reducing six local windows from 512 to 256 tokens increased training from 514.9M to 522.2M tokens without degrading val_bpb, indicating that the model can trade additional attention compute for throughput at this context scale.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "S"    # all local attention; final layer is forced to full context
>>>>>>> REPLACE