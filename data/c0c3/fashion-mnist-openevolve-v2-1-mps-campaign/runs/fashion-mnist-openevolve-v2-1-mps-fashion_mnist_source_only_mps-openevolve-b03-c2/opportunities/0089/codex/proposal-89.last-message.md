MECHANISM: Counter-directional center-weight refinement

HYPOTHESIS: Reducing the center-view weight from 3.25 to 3.125 will correct at least one ensemble error and raise validation-correct above 9,256 while retaining most of the center view’s stabilizing influence.

INTENDED_EDIT: Set the center-to-shift ensemble ratio to 3.125:1 and update its normalization constant, preserving the best verified temperature.

EVIDENCE: Raising the center weight to 3.5 reduced validation-correct from 9,256 to 9,253; testing a smaller counter-directional step is the most direct probe of whether lower center weighting improves the primary objective, while avoiding the unverified larger move to 3.0.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.125 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.25)
>>>>>>> REPLACE