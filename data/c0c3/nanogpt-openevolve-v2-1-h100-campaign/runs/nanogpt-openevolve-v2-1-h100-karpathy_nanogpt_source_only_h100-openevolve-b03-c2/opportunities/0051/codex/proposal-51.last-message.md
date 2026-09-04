MECHANISM: Moderately front-loaded local context with late-layer pruning

HYPOTHESIS: Using 448-token windows before the intermediate full-context layer and 256-token windows afterward will reduce val_bpb below 0.983618 by interpolating between the near-best 512/256 allocation and the higher-throughput 384/256 allocation.

INTENDED_EDIT: Set the three early local-attention layers to 448 tokens and the three late local-attention layers to 256 tokens, retaining full context at layers 4 and 8 and all optimizer settings.

EVIDENCE: The 512/256 design nearly tied the best result at 0.983620, while 384/256 reached 0.983738 with greater token exposure; 448/256 directly tests the midpoint of their early-context tradeoff while preserving the consistently favorable 256-token late windows.

<<<<<<< SEARCH
        long_window = config.sequence_len
        early_short_window = long_window // 8
        late_short_window = long_window // 4
=======
        long_window = config.sequence_len
        early_short_window = 7 * long_window // 32
        late_short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows widen from 256 to 512
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 448 to 256
>>>>>>> REPLACE