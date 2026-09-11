MECHANISM: Moderate-stacked test-time augmentation with stronger logit calibration

HYPOTHESIS: Grouping two spatial views and their mirrors per forward pass will complete within the verification limit while preserving the ten-view ensemble, and scaling its log-probabilities by 1.20 will preserve 9,252 correct predictions while reducing cross-entropy below 0.2235582.

INTENDED_EDIT: Replace five pair-batched evaluation passes with three moderately sized grouped passes and raise the inference-only ensemble scale from 1.10 to 1.20.

EVIDENCE: Pair batching completed successfully at scale 1.10 with 9,252 correct and 0.2235582 cross-entropy, while the more aggressive five-view stacking timed out; two-view grouping tests an intermediate throughput/memory tradeoff, and positive scaling cannot change the ensemble argmax.

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
        log_probabilities = []
        for start in range(0, len(views), 2):
            view_group = views[start : start + 2]
            stacked_views = torch.cat(view_group, dim=0)
            augmented_views = torch.cat(
                (stacked_views, stacked_views.flip(-1)), dim=0
            )
            grouped_log_probabilities = F.log_softmax(
                self._forward_once(augmented_views), dim=1
            )
            log_probabilities.extend(
                grouped_log_probabilities.chunk(2 * len(view_group), dim=0)
            )
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
        return 1.20 * ensemble_log_probabilities
>>>>>>> REPLACE