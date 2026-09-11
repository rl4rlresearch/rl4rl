MECHANISM: Augmentation-prior-weighted test-time ensemble

HYPOTHESIS: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by matching the training augmentation distribution.

INTENDED_EDIT: Replace uniform ten-view probability and geometric averaging with center/shift weights matching the exact training offset probabilities, while retaining flip averaging, ensemble blending, and calibration.

EVIDENCE: The verified 9,253-correct design weights all offsets equally although training samples center 5/13 and each cardinal shift 2/13; prior tests timed out without contradictory validation evidence.

<<<<<<< SEARCH
        stacked_log_probabilities = torch.stack(log_probabilities, dim=0)
        probability_ensemble = torch.logsumexp(
            stacked_log_probabilities, dim=0
        ) - math.log(len(log_probabilities))
        geometric_ensemble = stacked_log_probabilities.mean(dim=0)
=======
        stacked_log_probabilities = torch.stack(log_probabilities, dim=0)
        ensemble_weights = stacked_log_probabilities.new_tensor(
            (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        ) / 26.0
        broadcast_weights = ensemble_weights[:, None, None]
        probability_ensemble = torch.logsumexp(
            stacked_log_probabilities + broadcast_weights.log(), dim=0
        )
        geometric_ensemble = (
            stacked_log_probabilities * broadcast_weights
        ).sum(dim=0)
>>>>>>> REPLACE