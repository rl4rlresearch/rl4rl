MECHANISM: Receptive-field-stacked local attention with a single final global layer

HYPOTHESIS: Using 512-token attention in seven layers and full-context attention only in the final layer will process more than 513.3M tokens while preserving sequence-wide integration, reducing val_bpb below 0.993287.

INTENDED_EDIT: Change the depth-8 window pattern from SSSLSSSL to SSSSSSSL, retaining the proven 512-token short window and forced full-context final layer.

EVIDENCE: The 512-token design achieved the best result, 0.993287 val_bpb at 513.3M tokens; three consecutive 512-token layers already span much of the 2048-token sequence, so removing the intermediate global layer tests whether its compute is better spent on additional training tokens.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # 512-token local attention; final layer is forced full-context
>>>>>>> REPLACE