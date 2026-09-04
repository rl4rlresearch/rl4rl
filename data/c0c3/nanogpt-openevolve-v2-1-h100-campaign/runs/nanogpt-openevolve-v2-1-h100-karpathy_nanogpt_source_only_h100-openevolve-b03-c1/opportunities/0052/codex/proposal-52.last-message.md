MECHANISM: Two-channel attention-write routing bottleneck

HYPOTHESIS: Reducing each context gate from four to two input channels will lower val_bpb below 0.982693 by further suppressing noisy routing features while retaining token-conditional per-head write control.

INTENDED_EDIT: Halve only the contiguous hidden-state prefix used by every attention context gate, preserving the eight-channel value gates and all other architecture, optimizer, and schedule settings.

EVIDENCE: Narrowing context gates from eight to four channels improved val_bpb from 0.983164 to 0.982693, while narrower value-routing bottlenecks also consistently improved prior designs; this motivates one further controlled halving.

<<<<<<< SEARCH
        self.context_gate_channels = 4
=======
        self.context_gate_channels = 2
>>>>>>> REPLACE