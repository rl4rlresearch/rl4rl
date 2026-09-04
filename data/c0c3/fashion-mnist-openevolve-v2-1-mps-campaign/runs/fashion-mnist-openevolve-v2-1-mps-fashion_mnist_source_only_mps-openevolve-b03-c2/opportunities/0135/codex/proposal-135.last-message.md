MECHANISM: Midpoint search for the center-view weighting boundary

HYPOTHESIS: A center-view weight of 3.40625 will preserve all 9,256 correct predictions while lowering cross-entropy below 0.2144234748840332.

INTENDED_EDIT: Move halfway between the successful 3.375 and accuracy-losing 3.4375 center weights, update normalization, and use the strongest center-weighted reference’s calibration.

EVIDENCE: Weight 3.375 retained 9,256 correct at 0.21442347 cross-entropy, while 3.4375 reduced cross-entropy to 0.21441012 but lost two correct predictions; their midpoint efficiently probes the accuracy boundary.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
=======
            log_weight = math.log(3.40625 if view_index == 0 else 1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.35641 * aggregate_logits
=======
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.8125)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE