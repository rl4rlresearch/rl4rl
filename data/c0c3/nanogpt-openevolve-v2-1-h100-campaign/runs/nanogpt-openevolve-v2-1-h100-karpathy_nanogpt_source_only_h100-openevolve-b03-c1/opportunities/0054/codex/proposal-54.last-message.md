MECHANISM: Intermediate-width attention-write routing bottleneck

HYPOTHESIS: Increasing context-gate width from four to six channels will lower val_bpb below 0.982693 by approaching the estimated optimum between the regressive two- and eight-channel designs.

INTENDED_EDIT: Increase only each attention context gate’s contiguous hidden-state input from four channels to six, preserving all other architecture and training settings.

EVIDENCE: Four-channel gates achieved 0.982693 versus 0.983164 at eight channels and 0.984310 at two channels; the unverified six-channel attempt produced no contradictory metric, so the evidence-supported intermediate remains untested.

<<<<<<< SEARCH
        self.context_gate_channels = 4
=======
        self.context_gate_channels = 6
>>>>>>> REPLACE