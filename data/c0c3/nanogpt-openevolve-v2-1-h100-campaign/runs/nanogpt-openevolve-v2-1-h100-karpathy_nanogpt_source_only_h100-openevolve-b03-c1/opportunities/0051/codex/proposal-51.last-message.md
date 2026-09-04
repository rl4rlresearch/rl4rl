MECHANISM: Four-channel attention-write routing bottleneck

HYPOTHESIS: Reducing each context gate from eight to four input channels will lower val_bpb below 0.983164 by suppressing noisy routing features while retaining token-conditional per-head write control.

INTENDED_EDIT: Halve only the contiguous hidden-state prefix used by every attention context gate, preserving the winning value gates, architecture, optimizer, and schedule.

EVIDENCE: Eight-channel context gates improved val_bpb from 0.983900 to 0.983164, while value-routing gates improved as their bottleneck narrowed from 32 to 16 to 8 channels; this motivates testing whether context routing benefits from the same stronger bottleneck.

<<<<<<< SEARCH
        self.context_gate_channels = 8
        self.context_gate = nn.Linear(self.context_gate_channels, self.n_head, bias=False)
=======
        self.context_gate_channels = 4
        self.context_gate = nn.Linear(self.context_gate_channels, self.n_head, bias=False)
>>>>>>> REPLACE