MECHANISM: Hierarchical quarter-context sliding attention

HYPOTHESIS: Reducing short-layer attention from 1024 to 512 tokens while retaining full-context layers every fourth block will increase tokens trained in five minutes and lower val_bpb below 0.995558.

INTENDED_EDIT: Change only the short attention window from half to one quarter of the 2048-token sequence.

EVIDENCE: The depth-8 baseline reached val_bpb 0.995558 on 497.0M tokens at 39.58% MFU; six of eight layers use short attention, so reducing their window targets a substantial recurring cost while preserving two full-context layers.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE