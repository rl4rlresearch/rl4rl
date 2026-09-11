MECHANISM: Fine-grained upper-bracket local-attention refinement

HYPOTHESIS: A 428-token short window will reduce val_bpb below 0.983635 by preserving slightly more context than the best 424-token design without reaching the regression observed at 432 tokens.

INTENDED_EDIT: Increase all six short-attention windows from 424 to 428 tokens while preserving both full-context layers and every other setting.

EVIDENCE: The 424-token window achieved the best val_bpb of 0.983635, while 432 tokens regressed to 0.983917; 428 tokens bisects the remaining unexplored upper-side bracket.

<<<<<<< SEARCH
        short_window = 53 * long_window // 256
=======
        short_window = 107 * long_window // 512
>>>>>>> REPLACE