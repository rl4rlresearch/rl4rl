MECHANISM: Wider local-context refinement

HYPOTHESIS: Doubling local attention windows from 256 to 512 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.984467 by improving contextual feature construction without adding another costly global layer.

INTENDED_EDIT: Change short attention windows from one-eighth to one-quarter of the 2048-token sequence length.

EVIDENCE: The 128-token-window experiment regressed, indicating that local context was already capacity-limiting, while adding a third full-context layer also regressed to 0.985972; widening the six local layers tests additional context while preserving the best two-global-layer schedule.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 4
>>>>>>> REPLACE