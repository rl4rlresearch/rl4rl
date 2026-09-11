MECHANISM: Bracketed short-attention window refinement

HYPOTHESIS: A 384-token short window will reduce val_bpb below 0.983911 by preserving more local context than 320 tokens while retaining most of its throughput advantage over the worse 512-token design.

INTENDED_EDIT: Increase the six short-attention layers from 320-token to 384-token windows while preserving the two full-context layers and all other settings.

EVIDENCE: Increasing the short window from 256 to 320 tokens improved val_bpb from 0.983919 to 0.983911, while 512 tokens was worse at 0.984293; 384 tokens tests the midpoint of the remaining bracket.

<<<<<<< SEARCH
        short_window = 5 * long_window // 32
=======
        short_window = 3 * long_window // 16
>>>>>>> REPLACE