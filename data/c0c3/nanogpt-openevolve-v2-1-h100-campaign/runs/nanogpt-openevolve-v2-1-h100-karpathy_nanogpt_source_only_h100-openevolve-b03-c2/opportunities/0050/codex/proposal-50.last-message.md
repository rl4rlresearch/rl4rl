MECHANISM: Late-layer context pruning

HYPOTHESIS: Using 384-token local attention before the intermediate full-context layer and 256-token attention afterward will reduce `val_bpb` below 0.983618 by retaining the stronger early-context allocation while increasing token exposure.

INTENDED_EDIT: Restore the best design’s 384-token early windows, narrow the three post-integration local layers to 256 tokens, and retain full context at layers 4 and 8.

EVIDENCE: Uniform 384-token attention achieved 0.983618, while the 512/256 front-loaded split essentially tied it at 0.983620 and outperformed the reverse 256/512 split at 0.984213; this indicates early local context is more valuable and motivates pruning only late-layer context.

<<<<<<< SEARCH
        early_short_window = long_window // 4
        late_short_window = 3 * long_window // 16
=======
        early_short_window = 3 * long_window // 16
        late_short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 512 to 384
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 384 to 256
>>>>>>> REPLACE