MECHANISM: Sub-quarter local attention with a final global anchor

HYPOTHESIS: Seven 384-token local-attention layers followed by one full-context layer will process more than 518.5M tokens while retaining full-sequence integration, lowering val_bpb below 0.993870.

INTENDED_EDIT: Reduce local windows from 1024 to 384 tokens and change the layout to seven local layers plus the forced full-context final layer.

EVIDENCE: The 512-token single-anchor design achieved the best result at 0.993870 and 518.5M tokens; removing its final global anchor worsened val_bpb to 1.001821, so this patch preserves that anchor while testing whether cheaper local layers improve fixed-time training.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 3 * long_window // 16
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # seven 384-token local layers, then one forced full-context anchor
>>>>>>> REPLACE