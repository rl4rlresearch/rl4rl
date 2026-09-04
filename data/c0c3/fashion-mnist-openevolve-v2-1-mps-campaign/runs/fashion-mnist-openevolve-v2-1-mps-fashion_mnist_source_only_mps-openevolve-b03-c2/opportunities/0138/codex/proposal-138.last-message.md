MECHANISM: Binary refinement of center-view ensemble weighting

HYPOTHESIS: A center-view weight of 3.39453125 will preserve 9,256 correct predictions while lowering validation cross-entropy below 0.2144217742919922.

INTENDED_EDIT: Move halfway between the verified 3.3828125 weight and the accuracy-losing 3.40625 weight, update the normalization constant, and use the best verified calibration multiplier.

EVIDENCE: Weight 3.3828125 retained 9,256 correct at 0.2144217743 cross-entropy, while 3.40625 lost one prediction at 0.2144167206; their midpoint is the most informative remaining probe of the correctness boundary.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.39453125 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.356 * aggregate_logits
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.7890625)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE