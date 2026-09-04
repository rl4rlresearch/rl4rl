MECHANISM: Intermediate local-attention context

HYPOTHESIS: A 384-token short-attention window will preserve more predictive context than 256 tokens while retaining most of its throughput advantage, lowering val_bpb below 0.993215.

INTENDED_EDIT: Set the six short-attention layers to three-sixteenths of the 2048-token sequence length while retaining both full-context layers.

EVIDENCE: Moving from 512 to 256 tokens gained only 7.3M tokens and changed val_bpb by just 0.000004, while 128 tokens sharply regressed to 1.008181; this brackets the likely context-throughput optimum between 256 and 512 tokens.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = 3 * long_window // 16
>>>>>>> REPLACE