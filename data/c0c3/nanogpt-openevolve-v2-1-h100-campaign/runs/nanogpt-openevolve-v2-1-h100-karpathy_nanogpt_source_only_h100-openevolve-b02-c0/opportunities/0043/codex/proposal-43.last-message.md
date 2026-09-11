MECHANISM: Upper-bracket short-attention window refinement

HYPOTHESIS: A 432-token short window will reduce val_bpb below 0.983765 by preserving more local context than 416 tokens without incurring the full throughput loss observed at 448 tokens.

INTENDED_EDIT: Increase the six short-attention layers from 416-token to 432-token windows while preserving both full-context layers and all other settings.

EVIDENCE: The 416-token window achieved the best val_bpb of 0.983765, while 400 tokens regressed to 0.983933 and 448 tokens regressed to 0.984134; 432 tokens bisects the remaining unexplored upper bracket.

<<<<<<< SEARCH
        short_window = 13 * long_window // 64
=======
        short_window = 27 * long_window // 128
>>>>>>> REPLACE