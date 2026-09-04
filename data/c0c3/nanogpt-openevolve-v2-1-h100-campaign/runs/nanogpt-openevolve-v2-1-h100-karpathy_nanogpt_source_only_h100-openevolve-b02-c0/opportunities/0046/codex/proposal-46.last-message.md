MECHANISM: Upper-bracket local-attention window refinement

HYPOTHESIS: A 424-token short window will reduce val_bpb below 0.983765 by preserving slightly more context than 416 tokens without incurring the regression observed at 432 tokens.

INTENDED_EDIT: Increase all six short-attention windows from 416 to 424 tokens while preserving both full-context layers and every other setting.

EVIDENCE: Uniform 416-token windows achieved the best val_bpb of 0.983765, while 432-token windows regressed to 0.983917; 424 tokens bisects the remaining unexplored upper bracket.

<<<<<<< SEARCH
        short_window = 13 * long_window // 64
=======
        short_window = 53 * long_window // 256
>>>>>>> REPLACE