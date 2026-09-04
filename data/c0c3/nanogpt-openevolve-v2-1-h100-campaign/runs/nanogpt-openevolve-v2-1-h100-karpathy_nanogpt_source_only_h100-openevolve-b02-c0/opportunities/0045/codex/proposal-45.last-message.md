MECHANISM: Lower-bracket short-attention window refinement

HYPOTHESIS: A 408-token short window will reduce val_bpb below 0.983765 by retaining slightly more throughput than 416 tokens while preserving more local context than the worse 400-token design.

INTENDED_EDIT: Change the six short-attention layers from 416-token to 408-token windows, preserving both full-context layers and all other settings.

EVIDENCE: Uniform 416-token windows achieved the best val_bpb of 0.983765, while 400 and 432 tokens regressed to 0.983933 and 0.983917; 408 tokens bisects the unexplored lower bracket closest to the current optimum.

<<<<<<< SEARCH
        short_window = 13 * long_window // 64
=======
        short_window = 51 * long_window // 256
>>>>>>> REPLACE