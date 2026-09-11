MECHANISM: Bracketed short-attention window refinement

HYPOTHESIS: A 448-token short window will reduce val_bpb below 0.983803 by preserving more local context than 384 tokens while retaining most of its throughput advantage over the worse 512-token design.

INTENDED_EDIT: Increase the six short-attention layers from 384-token to 448-token windows while preserving the two full-context layers and all optimizer and schedule settings.

EVIDENCE: Increasing the short window from 320 to 384 tokens improved val_bpb from 0.983911 to 0.983803, while 512 tokens was worse at 0.984293; 448 tokens tests the midpoint of the remaining bracket.

<<<<<<< SEARCH
        short_window = 3 * long_window // 16
=======
        short_window = 7 * long_window // 32
>>>>>>> REPLACE