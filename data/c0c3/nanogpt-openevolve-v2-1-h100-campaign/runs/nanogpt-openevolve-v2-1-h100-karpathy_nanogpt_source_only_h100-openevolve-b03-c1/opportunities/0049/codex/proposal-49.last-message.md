MECHANISM: Eight-channel value-routing bottleneck

HYPOTHESIS: Reducing each learned value gate from 16 to 8 input channels will lower val_bpb below 0.983916 by further suppressing noisy routing features while retaining input-dependent value modulation.

INTENDED_EDIT: Halve the contiguous hidden-state prefix used by each per-head value-embedding gate from 16 channels to 8, preserving all other architecture and optimization settings.

EVIDENCE: Narrowing the gate from 32 to 16 channels improved val_bpb from 0.984083 to 0.983916, while widening it to 64 or 512 channels regressed to 0.984472 and 0.984493; fixed gates regressed substantially to 0.986944, motivating another measured bottleneck reduction rather than removing learned routing.

<<<<<<< SEARCH
        self.ve_gate_channels = 16
=======
        self.ve_gate_channels = 8
>>>>>>> REPLACE