MECHANISM: Separable 3×3 translation test-time ensemble

HYPOTHESIS: Adding diagonal translations at weight 1/3.3960161209106445 will alter class rankings constructively and increase validation_correct above the 9,257 plateau.

INTENDED_EDIT: Extend the existing center-and-cardinal ensemble with four diagonal views, using the diagonal weight implied by a separable translation kernel while preserving existing center/cardinal weights.

EVIDENCE: Multiple logit-calibration refinements preserved exactly 9,257 correct predictions and tied at 0.2144188995361328 cross-entropy, so further temperature probes are unlikely to improve the primary objective; the established weighted shift ensemble motivates testing its missing diagonal components.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        log_prob_views = []
        for view_index, view in enumerate(views):
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_weight = math.log(3.3960161209106445 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.792032241821289)
=======
        center_weight = 3.3960161209106445
        diagonal_weight = 1.0 / center_weight
        weighted_views = (
            (images, center_weight),
            (padded[:, :, 0:28, 1:29], 1.0),
            (padded[:, :, 2:30, 1:29], 1.0),
            (padded[:, :, 1:29, 0:28], 1.0),
            (padded[:, :, 1:29, 2:30], 1.0),
            (padded[:, :, 0:28, 0:28], diagonal_weight),
            (padded[:, :, 0:28, 2:30], diagonal_weight),
            (padded[:, :, 2:30, 0:28], diagonal_weight),
            (padded[:, :, 2:30, 2:30], diagonal_weight),
        )
        log_prob_views = []
        for view, weight in weighted_views:
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_weight = math.log(weight)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        total_weight = 2.0 * sum(weight for _, weight in weighted_views)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(total_weight)
>>>>>>> REPLACE