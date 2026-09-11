MECHANISM: Fine-grained multi-head context factorization

HYPOTHESIS: Replacing four 128-dimensional attention heads with eight 64-dimensional heads, while restoring SSSL, will preserve projection parameters and approximate attention FLOPs but learn more distinct context-selection patterns, reducing val_bpb below 0.995558.

INTENDED_EDIT: Restore the proven SSSL context allocation and halve head width so the 512-dimensional model uses eight full Q/K/V heads instead of four.

EVIDENCE: SSSS provided essentially no throughput gain and worsened val_bpb from 0.995558 to 0.997009, while reducing independent KV heads produced 1.018396. This challenges the load-bearing assumption that a few wide attention maps are sufficient and tests greater attention-pattern diversity without the GQA design’s loss of total K/V width or parameters.

<<<<<<< SEARCH
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSS" # half context throughout, with the final layer forced to full context
=======
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # eight narrower heads provide more independent context maps at fixed width
WINDOW_PATTERN = "SSSL" # proven mix of half-context and full-context attention
>>>>>>> REPLACE