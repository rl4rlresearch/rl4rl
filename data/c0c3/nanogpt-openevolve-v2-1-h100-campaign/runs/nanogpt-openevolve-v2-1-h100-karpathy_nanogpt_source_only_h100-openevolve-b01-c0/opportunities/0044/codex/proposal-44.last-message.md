MECHANISM: Selective early-layer context interpolation

HYPOTHESIS: Giving layer 2 a 1024-token window will beat 0.984059 val_bpb by adding useful early context without the throughput cost that made full attention at layer 2 regress.

INTENDED_EDIT: Use 1024-token attention at layers 2 and 6, retain full attention at layers 4 and 8, and keep 512-token windows elsewhere.

EVIDENCE: Interpolating layer 6 from 512-token to 1024-token attention improved val_bpb from 0.984313 to 0.984059, while extending it fully to 2048 regressed; applying the same midpoint interpolation to layer 2 directly tests whether the unsuccessful fourth full-attention layer was over-compute rather than intrinsically misplaced.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SMSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layers 2 and 6
>>>>>>> REPLACE