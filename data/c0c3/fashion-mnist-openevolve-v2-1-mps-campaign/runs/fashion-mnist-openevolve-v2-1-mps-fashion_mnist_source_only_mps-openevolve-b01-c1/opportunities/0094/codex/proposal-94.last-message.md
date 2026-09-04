MECHANISM: Evaluation-only center-weighted crop aggregation

HYPOTHESIS: Applying separable 1:2:1 weights only to the central validation crops will exceed 9,247 correct predictions by improving ensemble quality without imposing the training bias that accompanied the jointly weighted result.

INTENDED_EDIT: Weight the central 3×3 validation logits with a normalized separable 1:2:1 kernel while leaving training and the full 5×5 ensemble unchanged.

EVIDENCE: Joint 1:2:1 weighting reached 9,246 correct but improved cross-entropy from 0.21609 to 0.21532; the evaluation-only variant timed out, so its ability to retain that ensemble benefit without biased crop sampling remains untested.

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
                        (2 if offset_y == 2 else 1)
                        * (2 if offset_x == 2 else 1)
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