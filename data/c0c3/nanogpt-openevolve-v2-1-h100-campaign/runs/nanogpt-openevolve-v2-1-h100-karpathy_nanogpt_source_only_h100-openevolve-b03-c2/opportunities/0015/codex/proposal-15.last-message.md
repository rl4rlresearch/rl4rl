MECHANISM: Symmetric local cooldown refinement

HYPOTHESIS: A 56% linear warmdown will beat the best observed val_bpb of 0.986676 by testing the unmeasured longer side of the apparent optimum at 55%.

INTENDED_EDIT: Begin linear cooldown at 44% training progress while retaining the proven architecture, optimizer, and 262K-token update geometry.

EVIDENCE: The 55% warmdown achieved the best val_bpb of 0.986676; 54% regressed to 0.987812 and 60% reached 0.987466, motivating a symmetric one-point test at 56% before abandoning this local schedule optimum.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.4    # linear refinement over the final 40% of training
=======
WARMDOWN_RATIO = 0.56   # linear refinement over the final 56% of training
>>>>>>> REPLACE