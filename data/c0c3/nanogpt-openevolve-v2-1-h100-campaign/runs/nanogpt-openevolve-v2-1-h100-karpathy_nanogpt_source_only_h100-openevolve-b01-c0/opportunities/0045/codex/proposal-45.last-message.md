MECHANISM: Nonfinal full-attention midpoint interpolation

HYPOTHESIS: Reducing layer 4 from 2048-token full attention to a 1024-token window will beat 0.984059 val_bpb by retaining substantial mid-range context while increasing training throughput.

INTENDED_EDIT: Use 1024-token attention at layers 4 and 6, retain full attention at layer 8, and keep 512-token windows elsewhere.

EVIDENCE: At layer 6, the 1024-token window achieved 0.984059, outperforming both the 512-token endpoint at 0.984313 and the 2048-token endpoint at 0.984244; applying the same interpolation to the other nonfinal full-attention layer tests whether that context-compute optimum generalizes.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSMSMSL" # 1024-token attention at layers 4 and 6; full attention at layer 8
>>>>>>> REPLACE