MECHANISM: Three-stage periodic global attention

HYPOTHESIS: Using full attention at layers 3, 6, and 8 while retaining 512-token windows elsewhere will beat 0.993324 val_bpb by improving long-range mixing without entering the inefficient 256-token regime.

INTENDED_EDIT: Change the repeating attention pattern from SSSL to SSL, increasing full-attention layers from two to three.

EVIDENCE: Reducing full-attention layers from two to one regressed val_bpb from 0.993324 to 1.002549 and reduced training tokens from 514.3M to 454.0M, motivating the adjacent test in the opposite direction.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSL"  # full attention at layers 3, 6, and 8; S uses quarter context
>>>>>>> REPLACE