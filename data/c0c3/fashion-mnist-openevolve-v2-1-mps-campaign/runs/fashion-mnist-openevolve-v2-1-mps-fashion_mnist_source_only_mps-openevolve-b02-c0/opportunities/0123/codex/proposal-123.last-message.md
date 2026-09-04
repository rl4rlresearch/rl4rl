MECHANISM: Augmentation-prior-weighted test-time ensemble

HYPOTHESIS: Weighting the center view 5/13 and each translated view 2/13 will exceed 9,253 correct predictions by reducing distortion from overrepresented shifted views.

INTENDED_EDIT: Match inference ensemble weights to the training augmentation’s exact center-versus-shift sampling probabilities while retaining flip averaging, ensemble blending, and calibration.

EVIDENCE: Calibration has converged near 1.3715955 with 9,253 predictions unchanged; training samples the center offset 5/13 of the time and each cardinal shift 2/13, whereas inference currently weights all five offsets equally.

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
        ).view(-1, 1, 1)
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        probability_ensemble = torch.logsumexp(
            stacked_log_probabilities + ensemble_weights.log(), dim=0
        )
        geometric_ensemble = (
            stacked_log_probabilities * ensemble_weights
        ).sum(dim=0)
>>>>>>> REPLACE