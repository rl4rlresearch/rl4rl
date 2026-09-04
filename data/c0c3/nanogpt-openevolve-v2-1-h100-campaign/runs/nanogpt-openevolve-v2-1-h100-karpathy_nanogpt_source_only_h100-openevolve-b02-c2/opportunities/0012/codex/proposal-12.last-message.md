MECHANISM: Bracketed cautious-decay refinement

HYPOTHESIS: Restoring NorMuon beta2 to 0.95 and setting Muon weight decay to 0.095 will lower val_bpb below 0.985746 by refining the apparent optimum near 0.1 while retaining the proven variance averaging.

INTENDED_EDIT: Restore the best-performing NorMuon beta2 and test a slightly lower cautious weight decay of 0.095 in the proven 262K-token configuration.

EVIDENCE: With beta2 0.95, weight decay 0.1 achieved 0.985746, outperforming 0.05 at 0.987328, 0.125 at 0.986461, and 0.2 at 0.987182; beta2 0.975 also worsened validation to 0.988196.

<<<<<<< SEARCH
                momentum=0.95, ns_steps=5, beta2=0.975, weight_decay=weight_decay,
=======
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
>>>>>>> REPLACE

<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.095    # refine the observed cautious-decay optimum near 0.1
>>>>>>> REPLACE