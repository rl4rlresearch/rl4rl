MECHANISM: Upper-bound refinement of center-view ensemble weighting

HYPOTHESIS: A center-view weight of 3.400390625 will preserve 9,257 correct predictions while lowering validation cross-entropy below 0.2144192337036133.

INTENDED_EDIT: Move halfway from the best verified weight, 3.39453125, toward the accuracy-losing 3.40625 weight; update normalization and use the verified 1.35605 calibration.

EVIDENCE: Weight 3.39453125 achieved the best result with 9,257 correct, while 3.40625 achieved 9,255 with slightly lower cross-entropy; their midpoint is the most informative probe of the upper accuracy boundary.

<<<<<<< SEARCH
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.35641 * aggregate_logits
=======
            log_weight = math.log(3.400390625 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.80078125)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE