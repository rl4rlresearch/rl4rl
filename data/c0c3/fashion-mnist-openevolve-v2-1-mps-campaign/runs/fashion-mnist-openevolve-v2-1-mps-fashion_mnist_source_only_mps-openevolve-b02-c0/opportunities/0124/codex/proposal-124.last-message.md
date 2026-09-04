MECHANISM: Augmentation-prior-weighted test-time ensemble

HYPOTHESIS: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by matching the training augmentation distribution and reducing distortion from overrepresented shifted views.

INTENDED_EDIT: Replace uniform averaging across the ten original/flipped inference views with probability and geometric ensembles weighted by the exact training offset probabilities, while retaining flip averaging, ensemble blending, and calibration.

EVIDENCE: Training selects the center offset with probability 5/13 and each cardinal shift with probability 2/13, but verified inference weights all offsets equally; the previous test of this targeted change timed out and supplied no contradictory score evidence.

<<<<<<< SEARCH
        stacked_log_probabilities = torch.stack(log_probabilities, dim=0)
        probability_ensemble = torch.logsumexp(
            stacked_log_probabilities, dim=0
        ) - math.log(len(log_probabilities))
        geometric_ensemble = stacked_log_probabilities.mean(dim=0)
=======
        stacked_log_probabilities = torch.stack(log_probabilities, dim=0)
        view_weights = stacked_log_probabilities.new_tensor(
            (5 / 26, 5 / 26) + (1 / 13,) * 8
        )[:, None, None]
        probability_ensemble = torch.logsumexp(
            stacked_log_probabilities + view_weights.log(), dim=0
        )
        geometric_ensemble = (
            view_weights * stacked_log_probabilities
        ).sum(dim=0)
>>>>>>> REPLACE