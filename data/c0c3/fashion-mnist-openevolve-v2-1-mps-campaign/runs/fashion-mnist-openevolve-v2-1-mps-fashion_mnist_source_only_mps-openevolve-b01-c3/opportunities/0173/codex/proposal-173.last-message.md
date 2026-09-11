MECHANISM: Moderately increased flip-offset orbit exposure

HYPOTHESIS: Increasing cross-offset reflected pairs from 25% to 37.5% while retaining 5% consistency will exceed 9,318 correct predictions by strengthening the beneficial translation constraint without moving fully to the unverified 50% mixture.

INTENDED_EDIT: Raise the stochastic cross-offset pair frequency to 37.5%; preserve the architecture, two-view compute, consistency strength, and inference ensemble.

EVIDENCE: Raising cross-offset consistency from 2.5% to 5% improved correctness from 9,312 to 9,318, showing that stronger translation alignment helps; 37.5% is a conservative midpoint between the validated 25% frequency and the 50% attempt that did not finish.

<<<<<<< SEARCH
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.25
=======
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
>>>>>>> REPLACE