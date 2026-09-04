MECHANISM: Balanced-class bias reallocation to refinement-gate capacity

HYPOTHESIS: Expanding the proven dual-statistic gate from seven to eight hidden channels while removing the dispensable ten-class output bias will exceed 9,322 correct predictions within the 250,000-parameter ceiling.

INTENDED_EDIT: Use an eight-channel refinement-gate bottleneck and disable the final classifier bias, yielding exactly 250,000 learned parameters.

EVIDENCE: Image-conditioned gating improved the ungated backbone from 9,315 to 9,319 correct, and the equal mean/max descriptor further reached 9,322; increasing capacity without altering that proven descriptor directly tests whether the gate remains bottlenecked.

<<<<<<< SEARCH
        gate_channels = max(channels // 9, 1)
=======
        gate_channels = max(channels // 8, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(56, 10),
=======
            nn.Linear(56, 10, bias=False),
>>>>>>> REPLACE