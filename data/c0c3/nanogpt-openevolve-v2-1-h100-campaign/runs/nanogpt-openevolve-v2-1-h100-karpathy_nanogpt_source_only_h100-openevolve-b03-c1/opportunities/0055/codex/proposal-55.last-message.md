MECHANISM: Four-channel value-residual routing bottleneck

HYPOTHESIS: Reducing value-gate inputs from eight to four channels will lower val_bpb below 0.982693 by further suppressing noisy routing features while retaining learned per-head value modulation.

INTENDED_EDIT: Halve only the contiguous hidden-state prefix used by value-embedding gates, preserving the proven four-channel context gates and all other settings.

EVIDENCE: Narrowing value gates from 32 to 16 to 8 channels improved val_bpb from 0.984083 to 0.983916 to 0.983900; the strong result from four-channel context gates motivates testing the same compact bottleneck for value routing.

<<<<<<< SEARCH
        self.ve_gate_channels = 8
=======
        self.ve_gate_channels = 4
>>>>>>> REPLACE