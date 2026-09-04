MECHANISM: Narrower local-attention windows with preserved global integration

HYPOTHESIS: Reducing short attention windows from 1024 to 768 tokens while retaining full-context layers 4 and 8 will increase throughput enough to lower val_bpb below 0.985746 without materially reducing usable context.

INTENDED_EDIT: Restore the best verified 50% linear warmdown and reduce only the six local-attention layers to three-eighths context.

EVIDENCE: Two full-context layers achieved 0.985746, outperforming one full layer at 0.986118 and three full layers at 0.986305; this motivates preserving the proven global-attention count while reducing local-attention compute.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 3 * long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin LR convergence earlier after shorter warmdown regressed
=======
WARMDOWN_RATIO = 0.5    # best verified linear warmdown
>>>>>>> REPLACE