MECHANISM: Canonical-orientation-biased test-time ensemble

HYPOTHESIS: A modest 1.10:0.90 original-to-mirrored weighting will correct at least one borderline ensemble prediction and raise validation_correct above 9,256; the verified 1.35605 multiplier will maintain near-optimal calibration.

INTENDED_EDIT: Preserve the validated 3.25:1 center-to-axis balance, bias every view toward its unmirrored orientation while keeping total ensemble weight unchanged, and apply the best verified temperature.

EVIDENCE: Both tested center-weight directions and added diagonal views reduced validation_correct, motivating an orthogonal refinement that leaves spatial-view weights intact; Reference Design 2 verified 1.35605 as the lowest-cross-entropy multiplier at the baseline weighting.

<<<<<<< SEARCH
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.30 * aggregate_logits
=======
            log_prob_views.extend(
                (
                    original + log_weight + math.log(1.10),
                    mirrored + log_weight + math.log(0.90),
                )
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE