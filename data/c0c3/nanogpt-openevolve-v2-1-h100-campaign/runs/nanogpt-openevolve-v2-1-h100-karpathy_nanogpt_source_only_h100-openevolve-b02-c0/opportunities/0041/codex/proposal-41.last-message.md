MECHANISM: Bracketed short-attention window refinement

HYPOTHESIS: A 416-token short window will reduce val_bpb below 0.983803 by preserving slightly more local context than 384 tokens without incurring the larger throughput loss observed at 448 tokens.

INTENDED_EDIT: Increase the six short-attention layers from 384-token to 416-token windows while preserving both full-context layers and all other settings.

EVIDENCE: The 384-token window improved val_bpb to 0.983803, while 320 tokens was slightly worse at 0.983911 and 448 tokens regressed to 0.984134; 416 tokens bisects the remaining upper-side bracket.

<<<<<<< SEARCH
        short_window = 3 * long_window // 16
=======
        short_window = 13 * long_window // 64
>>>>>>> REPLACE