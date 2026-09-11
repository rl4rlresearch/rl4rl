MECHANISM: Add a late intermediate global-attention layer

HYPOTHESIS: Three full-context layers at depths 3, 5, and 7 will reduce val_bpb below 0.983635 because the large regression from removing the intermediate global layer indicates that global mixing is worth more than its throughput cost.

INTENDED_EDIT: Preserve the 424-token local window while changing layer 5 from local to full-context attention, producing five local and three global layers.

EVIDENCE: Replacing the intermediate full-context layer with local attention increased throughput from 506.2M to 513.3M tokens but worsened val_bpb from 0.983635 to 0.984677, showing that additional global mixing can dominate the value of extra training tokens.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSLSLSL" # full context at layers 3, 5, and 7
>>>>>>> REPLACE