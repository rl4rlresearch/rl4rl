MECHANISM: Evaluation-only center-weighted crop aggregation

HYPOTHESIS: Applying the separable 1:2:1 central-crop weighting only during validation will exceed 9,247 correct predictions by retaining uniform central-crop training while capturing the stronger kernel’s demonstrated cross-entropy improvement.

INTENDED_EDIT: Weight the central 3×3 validation logits with a separable 1:2:1 kernel and normalize by its total weight, leaving training sampling and all learned components unchanged.

EVIDENCE: Jointly center-weighting training and validation with 1:2:1 reached 9,246 correct but improved cross-entropy from 0.21609 to 0.21532; isolating the evaluation-side change tests whether the lost prediction came from biased training rather than the better-calibrated ensemble.

<<<<<<< SEARCH
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    if central_logits_sum is None:
                        central_logits_sum = view_logits
                    else:
                        central_logits_sum = central_logits_sum + view_logits
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 18.0
=======
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    central_weight = (
                        (2.0 if offset_y == 2 else 1.0)
                        * (2.0 if offset_x == 2 else 1.0)
                    )
                    if central_logits_sum is None:
                        central_logits_sum = central_weight * view_logits
                    else:
                        central_logits_sum = (
                            central_logits_sum
                            + central_weight * view_logits
                        )
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 32.0
>>>>>>> REPLACE