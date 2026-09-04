MECHANISM: Interpolated local-attention compute allocation

HYPOTHESIS: Using 192-token local-attention windows will achieve `val_bpb < 0.982793` by balancing the quality advantage of 256-token windows with the higher throughput observed at 128 tokens.

INTENDED_EDIT: Set the six `S` layers to a 3/32-context (192-token) attention window while retaining two full-context layers and all other settings.

EVIDENCE: Moving from 512 to 256 tokens improved `val_bpb` from `0.983023` to `0.982793`, while 128 tokens increased throughput but slightly regressed to `0.982806`; quadratic interpolation over the 512/256/128 results places the estimated optimum near 192 tokens.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = 3 * long_window // 32
>>>>>>> REPLACE