MECHANISM: Augmentation-distribution-matched ensemble weighting

HYPOTHESIS: Weighting the center view at 5/13 and each shifted view at 2/13, matching their training-time sampling frequencies, will increase validation_correct above 9,252 while retaining the successful 1.10 calibration scale.

INTENDED_EDIT: Replace equal weighting of the ten test-time views with probability-space weights matching the spatial augmentation distribution; mirrored and unmirrored variants split each spatial weight equally.

EVIDENCE: The best design uses equal-weight ten-view averaging, but its training augmentation samples the center crop 5/13 of the time and each directional shift 2/13; aligning inference weights with this observed training distribution is the most direct accuracy-oriented change.

<<<<<<< SEARCH
        log_probabilities = []
        for view in views:
            paired_views = torch.cat((view, view.flip(-1)), dim=0)
            paired_log_probabilities = F.log_softmax(
                self._forward_once(paired_views), dim=1
            )
            log_probabilities.extend(paired_log_probabilities.chunk(2, dim=0))
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return 1.10 * ensemble_log_probabilities
=======
        spatial_weights = (5.0, 2.0, 2.0, 2.0, 2.0)
        weighted_log_probabilities = []
        for view, spatial_weight in zip(views, spatial_weights):
            paired_views = torch.cat((view, view.flip(-1)), dim=0)
            paired_log_probabilities = F.log_softmax(
                self._forward_once(paired_views), dim=1
            )
            log_weight = math.log(spatial_weight / 26.0)
            weighted_log_probabilities.extend(
                log_probabilities + log_weight
                for log_probabilities in paired_log_probabilities.chunk(2, dim=0)
            )
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(weighted_log_probabilities, dim=0), dim=0
        )
        return 1.10 * ensemble_log_probabilities
>>>>>>> REPLACE