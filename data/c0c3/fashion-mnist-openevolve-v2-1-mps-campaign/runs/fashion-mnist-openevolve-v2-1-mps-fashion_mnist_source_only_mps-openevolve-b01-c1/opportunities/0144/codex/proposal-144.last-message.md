MECHANISM: Near-arithmetic log-opinion pooling

HYPOTHESIS: Blending 1% of the prior geometric logit ensemble into arithmetic probability pooling will retain 9,266 correct predictions while lowering cross-entropy below 0.215801, exceeding validation_score 9266.411252.

INTENDED_EDIT: Accumulate validation-view logits alongside probabilities and interpolate 99% of the calibrated arithmetic log-probabilities with 1% of the geometric ensemble’s normalized logits.

EVIDENCE: Arithmetic pooling improved correct predictions from 9,265 to 9,266 but worsened cross-entropy from 0.211944 to 0.215801; a small interpolation toward the better-calibrated geometric endpoint is likely to improve the tie-breaker without crossing enough decision boundaries to lose the accuracy gain.

<<<<<<< SEARCH
        probability_sum = None
        central_probability_sum = None
=======
        probability_sum = None
        central_probability_sum = None
        logit_sum = None
        central_logit_sum = None
>>>>>>> REPLACE

<<<<<<< SEARCH
                probabilities = F.softmax(logits, dim=-1)
                original_probs, flipped_probs = probabilities.chunk(2, dim=0)
                view_probabilities = original_probs + flipped_probs
                if probability_sum is None:
                    probability_sum = view_probabilities
                else:
                    probability_sum = probability_sum + view_probabilities
=======
                probabilities = F.softmax(logits, dim=-1)
                original_probs, flipped_probs = probabilities.chunk(2, dim=0)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_probabilities = original_probs + flipped_probs
                view_logits = original_logits + flipped_logits
                if probability_sum is None:
                    probability_sum = view_probabilities
                    logit_sum = view_logits
                else:
                    probability_sum = probability_sum + view_probabilities
                    logit_sum = logit_sum + view_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
                    if central_probability_sum is None:
                        central_probability_sum = (
                            central_weight * view_probabilities
                        )
                    else:
                        central_probability_sum = (
                            central_probability_sum
                            + central_weight * view_probabilities
                        )
=======
                    if central_probability_sum is None:
                        central_probability_sum = (
                            central_weight * view_probabilities
                        )
                        central_logit_sum = central_weight * view_logits
                    else:
                        central_probability_sum = (
                            central_probability_sum
                            + central_weight * view_probabilities
                        )
                        central_logit_sum = (
                            central_logit_sum
                            + central_weight * view_logits
                        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_ensemble = probability_sum / 50.0
        central_ensemble = central_probability_sum / 32.0
        ensemble = 0.9 * full_ensemble + 0.1 * central_ensemble
        return 1.29834 * ensemble.clamp_min(1.0e-7).log()
=======
        full_ensemble = probability_sum / 50.0
        central_ensemble = central_probability_sum / 32.0
        ensemble = 0.9 * full_ensemble + 0.1 * central_ensemble
        arithmetic_logits = 1.29834 * ensemble.clamp_min(1.0e-7).log()
        geometric_logits = (
            0.9 * logit_sum / 50.0
            + 0.1 * central_logit_sum / 32.0
        )
        geometric_logits = F.log_softmax(geometric_logits, dim=-1)
        return 0.99 * arithmetic_logits + 0.01 * geometric_logits
>>>>>>> REPLACE