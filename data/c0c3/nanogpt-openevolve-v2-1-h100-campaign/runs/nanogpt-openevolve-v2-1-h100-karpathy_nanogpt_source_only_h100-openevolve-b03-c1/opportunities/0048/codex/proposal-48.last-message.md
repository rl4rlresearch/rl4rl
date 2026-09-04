MECHANISM: Tighter value-routing bottleneck

HYPOTHESIS: Reducing the learned value-gate input from 32 to 16 channels will lower val_bpb below 0.984083 by suppressing noisy routing features while retaining the learned modulation whose removal caused a large regression.

INTENDED_EDIT: Halve the contiguous hidden-state prefix used by each per-head value-embedding gate, without changing its optimizer, initialization, or the winning attention-window pattern.

EVIDENCE: Learned 32-channel gates achieved 0.984083, while widening them to 64 or 512 channels regressed to 0.984472 and 0.984493; fixed gates regressed much further to 0.986944, motivating a narrower learned bottleneck rather than removing routing.

<<<<<<< SEARCH
        self.ve_gate_channels = 32
=======
        self.ve_gate_channels = 16
>>>>>>> REPLACE