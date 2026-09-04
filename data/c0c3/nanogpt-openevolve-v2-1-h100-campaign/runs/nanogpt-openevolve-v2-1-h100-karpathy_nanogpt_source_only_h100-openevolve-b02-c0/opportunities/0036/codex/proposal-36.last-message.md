MECHANISM: Bracketed short-attention window refinement

HYPOTHESIS: A 192-token short window will reduce val_bpb below 0.983919 by preserving more local context than the regressing 128-token design while retaining most of the compute savings achieved at 256 tokens.

INTENDED_EDIT: Change the six short-attention layers from 256-token to 192-token windows; preserve the two full-context layers and all other architecture, optimizer, and schedule settings.

EVIDENCE: Reducing short attention from 512 to 256 tokens improved val_bpb from 0.984293 to 0.983919, but reducing it further to 128 tokens regressed to 0.984000; 192 tokens directly refines the bracket around the current optimum.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = 3 * long_window // 32
>>>>>>> REPLACE