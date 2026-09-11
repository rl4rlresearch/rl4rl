MECHANISM: Quadratic cautious-decay refinement

HYPOTHESIS: Setting Muon weight decay to 0.11 will lower val_bpb below 0.985746 by moving toward the interpolated minimum implied by the tested decay values.

INTENDED_EDIT: Keep the proven 262K-token configuration and 50% LR warmdown unchanged while changing cautious Muon weight decay from 0.125 to 0.11.

EVIDENCE: Weight decay 0.1 achieved 0.985746, outperforming 0.095 at 0.986110 and 0.125 at 0.986461; these three measurements suggest an interior optimum slightly above 0.1.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.125    # interpolate near the observed cautious-decay optimum
=======
WEIGHT_DECAY = 0.11     # refine the bracketed cautious-decay optimum slightly above 0.1
>>>>>>> REPLACE