MECHANISM: Low-weight diagonal test-time augmentation

HYPOTHESIS: Adding four diagonal one-pixel views at 0.25 weight each will provide complementary translation evidence and increase validation_correct above 9,256.

INTENDED_EDIT: Extend the verified center-and-axis ensemble with four lightly weighted diagonal views, update normalization, and use the best verified temperature multiplier.

EVIDENCE: Changing the center weight in either tested direction reduced validation_correct, indicating that the 3.25:1 center-to-axis balance should be preserved; diagonal views add new evidence without disturbing those established weights.

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
            log_weight = math.log(3.25 if view_index == 0 else 1.0)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(14.5)
        return 1.33 * aggregate_logits
=======
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
            padded[:, :, 0:28, 0:28],
            padded[:, :, 0:28, 2:30],
            padded[:, :, 2:30, 0:28],
            padded[:, :, 2:30, 2:30],
        )
        view_weights = (3.25, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25)
        log_prob_views = []
        for view, view_weight in zip(views, view_weights):
            paired = torch.cat((view, view.flip(-1)), dim=0)
            original, mirrored = F.log_softmax(
                self._forward_once(paired), dim=1
            ).chunk(2, dim=0)
            log_weight = math.log(view_weight)
            log_prob_views.extend(
                (original + log_weight, mirrored + log_weight)
            )

        stacked = torch.stack(log_prob_views, dim=0)
        aggregate_logits = torch.logsumexp(stacked, dim=0) - math.log(16.5)
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE