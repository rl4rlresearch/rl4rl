MECHANISM: Bracketed short-attention window refinement

HYPOTHESIS: A 320-token short window will reduce val_bpb below 0.983919 by preserving more local context than 256 tokens while retaining most of its throughput advantage over 512 tokens.

INTENDED_EDIT: Increase the six short-attention layers from 256-token to 320-token windows while preserving the two full-context layers and all optimizer and schedule settings.

EVIDENCE: The 256-token window achieved the best val_bpb of 0.983919; reducing it to 192 or 128 tokens regressed, while 512 tokens was also worse, motivating a conservative test on the unexplored interval between 256 and 512.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = 5 * long_window // 32
>>>>>>> REPLACE