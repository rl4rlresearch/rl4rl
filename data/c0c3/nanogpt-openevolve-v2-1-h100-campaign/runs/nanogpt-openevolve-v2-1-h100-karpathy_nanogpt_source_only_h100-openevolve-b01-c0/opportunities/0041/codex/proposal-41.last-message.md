MECHANISM: Symmetric layer-6 context interpolation

HYPOTHESIS: A 1280-token layer-6 window will beat 0.984059 val_bpb by adding useful context beyond 1024 without incurring the throughput cost of full attention.

INTENDED_EDIT: Increase only layer 6’s medium attention window from 1024 to 1280 tokens; layers 4 and 8 remain full-attention and all other layers remain at 512 tokens.

EVIDENCE: The 1024-token window achieved the best 0.984059 val_bpb; 768 and 2048 both regressed, so 1280 tests the equally spaced upper neighbor around the observed optimum.

<<<<<<< SEARCH
        medium_window = long_window // 2
=======
        medium_window = 5 * long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1280-token attention at layer 6
>>>>>>> REPLACE