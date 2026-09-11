MECHANISM: Intermediate-width attention-write routing bottleneck

HYPOTHESIS: Expanding each context gate from four to six input channels will lower val_bpb below 0.982693 by approaching the approximately six-channel optimum interpolated from the measured two-, four-, and eight-channel results.

INTENDED_EDIT: Increase only the contiguous hidden-state prefix used by every attention context gate from four channels to six, preserving the eight-channel value gates and all other settings.

EVIDENCE: Four-channel context gates achieved 0.982693, outperforming eight channels at 0.983164, while narrowing further to two channels regressed sharply to 0.984310; this non-monotonic response motivates testing the intermediate six-channel width.

<<<<<<< SEARCH
        self.context_gate_channels = 4
        self.context_gate = nn.Linear(self.context_gate_channels, self.n_head, bias=False)
=======
        self.context_gate_channels = 6
        self.context_gate = nn.Linear(self.context_gate_channels, self.n_head, bias=False)
>>>>>>> REPLACE