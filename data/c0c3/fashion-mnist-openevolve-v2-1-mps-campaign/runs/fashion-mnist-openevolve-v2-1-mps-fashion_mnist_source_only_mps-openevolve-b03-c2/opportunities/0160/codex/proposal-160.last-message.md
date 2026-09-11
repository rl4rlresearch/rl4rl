MECHANISM: Symmetric logit-temperature refinement

HYPOTHESIS: Restoring the best verified center-view weight and lowering the positive logit scale to 1.35604 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891098022461.

INTENDED_EDIT: Restore center weight 3.3960161209106445 with exact normalization and probe the opposite calibration direction from the unsuccessful 1.35606 scale.

EVIDENCE: Scale 1.35605 achieved the best verified cross-entropy, while increasing it to 1.35606 worsened cross-entropy without changing accuracy; an equally sized downward probe is the most informative unresolved calibration test.

<<<<<<< SEARCH
            log_weight = math.log(3.396007537841797 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792015075683594)
        return 1.35605 * aggregate_logits
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
        return 1.35604 * aggregate_logits
>>>>>>> REPLACE