MECHANISM: Bracketed local-attention window interpolation

HYPOTHESIS: Seven 192-token local-attention layers followed by one full-context layer will preserve the throughput benefit of short windows while avoiding the quality loss observed at 128 tokens, lowering val_bpb below 0.992854.

INTENDED_EDIT: Reduce the seven local-attention windows from 384 to 192 tokens while preserving the final full-context anchor and all other architecture, optimizer, batching, and schedule settings.

EVIDENCE: Validation improved as local windows shrank from 512 to 384 to 256 tokens (0.993870 to 0.993518 to 0.992854), but regressed at 128 tokens to 0.995793; testing 192 tokens directly probes the bracketed optimum between 128 and 256.

<<<<<<< SEARCH
        short_window = 3 * long_window // 16
=======
        short_window = 3 * long_window // 32
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 384-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 192-token local layers, then one forced full-context anchor
>>>>>>> REPLACE