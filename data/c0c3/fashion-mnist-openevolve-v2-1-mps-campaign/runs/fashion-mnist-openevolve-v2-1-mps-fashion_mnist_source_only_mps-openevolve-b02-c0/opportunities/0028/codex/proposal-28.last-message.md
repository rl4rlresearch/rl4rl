MECHANISM: Geometric-mean cardinal-view ensemble

HYPOTHESIS: Averaging logits across the same cardinal and flipped views will exceed 9,249 correct predictions by rewarding cross-view class agreement instead of allowing one high-probability view to dominate the arithmetic probability ensemble.

INTENDED_EDIT: Replace probability-space averaging with logit-space averaging while reusing the already-computed centered logits, preserving all ten inference views and the training procedure.

EVIDENCE: Cardinal-only inference improved validation_correct from 9,238 to 9,249, confirming these views contain useful complementary evidence; spatial probability reweighting reduced accuracy, so changing only the aggregation rule is the most targeted remaining inference test.

<<<<<<< SEARCH
        logits = self._forward_once(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        height, width = images.shape[-2:]
        views = (
            images,
            padded[..., :height, 1 : width + 1],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, 1 : width + 1],
        )
        log_probabilities = []
        for view in views:
            log_probabilities.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_probabilities.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
        logits = self._forward_once(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        height, width = images.shape[-2:]
        shifted_views = (
            padded[..., :height, 1 : width + 1],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, 1 : width + 1],
        )
        view_logits = [
            logits,
            self._forward_once(images.flip(-1)),
        ]
        for view in shifted_views:
            view_logits.append(self._forward_once(view))
            view_logits.append(self._forward_once(view.flip(-1)))
        return torch.stack(view_logits, dim=0).mean(dim=0)
>>>>>>> REPLACE