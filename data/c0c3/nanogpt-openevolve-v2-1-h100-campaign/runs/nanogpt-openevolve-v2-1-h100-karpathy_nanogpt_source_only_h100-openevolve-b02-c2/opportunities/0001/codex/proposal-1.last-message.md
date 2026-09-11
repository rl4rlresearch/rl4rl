MECHANISM: Hierarchical 512-token local attention

HYPOTHESIS: Reducing short-window attention from 1024 to 512 tokens will lower val_bpb below 0.995558 by increasing training throughput while preserving full-context mixing in layers 4 and 8.

INTENDED_EDIT: Change the six short-attention layers to quarter-context windows; the two long-attention layers remain unchanged.

EVIDENCE: The baseline processes 497.0M tokens at 39.58% MFU with six of eight layers using 1024-token attention, so local attention remains a substantial fixed-time compute cost.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE