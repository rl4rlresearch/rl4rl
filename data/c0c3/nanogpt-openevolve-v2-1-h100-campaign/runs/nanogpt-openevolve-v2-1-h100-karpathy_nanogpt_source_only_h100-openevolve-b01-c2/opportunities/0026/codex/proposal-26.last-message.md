MECHANISM: Above-optimum local-attention window refinement

HYPOTHESIS: Seven 272-token local-attention layers followed by the full-context final anchor will retain at least 523M training tokens and lower val_bpb below 0.992854.

INTENDED_EDIT: Increase the seven local-attention windows from 224 to 272 tokens while preserving the architecture, optimizer, batching, schedule, and final global anchor.

EVIDENCE: The 256-token design achieved the best val_bpb of 0.992854, while 224 tokens reached 0.992930 and 384 tokens regressed to 0.993518; 272 tokens is the nearest untested refinement above the observed optimum.

<<<<<<< SEARCH
        short_window = 7 * long_window // 64
=======
        short_window = 17 * long_window // 128
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 224-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 272-token local layers, then one forced full-context anchor
>>>>>>> REPLACE