MECHANISM: Local-attention window interpolation

HYPOTHESIS: A 400-token short window will reduce val_bpb below 0.983765 by retaining more throughput than 416 tokens while preserving more local context than the slightly worse 384-token design.

INTENDED_EDIT: Change the six short-attention layers from 416-token to 400-token windows, preserving both full-context layers and all other settings.

EVIDENCE: The 416-token window achieved 0.983765, improving on 384 tokens at 0.983803, while 448 tokens regressed to 0.984134; 400 tokens refines the lower half of the remaining bracket around the current optimum.

<<<<<<< SEARCH
        short_window = 13 * long_window // 64
=======
        short_window = 25 * long_window // 128
>>>>>>> REPLACE