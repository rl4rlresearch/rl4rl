MECHANISM: Lean center-view flip ensemble

HYPOTHESIS: Reducing evaluation from ten augmented views to the center view and its mirror will complete verification reliably while retaining at least 9,200 correct predictions and comparable calibration.

INTENDED_EDIT: Replace the five-pass ten-view ensemble with one pair-batched forward pass over the original images and their horizontal mirrors, retaining the verified 1.10 scale.

EVIDENCE: The ten-view pair-batched design reached 9,252 correct, but every subsequent design timed out, including a six-view variant; cutting evaluation workload from 100,000 to 20,000 image views directly addresses the recurring completion failure while preserving the training-aligned flip ensemble.

<<<<<<< SEARCH
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
        paired_images = torch.cat((images, images.flip(-1)), dim=0)
        paired_log_probabilities = F.log_softmax(
            self._forward_once(paired_images), dim=1
        )
        original_log_probabilities, mirrored_log_probabilities = (
            paired_log_probabilities.chunk(2, dim=0)
        )
        ensemble_log_probabilities = torch.logsumexp(
            torch.stack(
                (original_log_probabilities, mirrored_log_probabilities), dim=0
            ),
            dim=0,
        ) - math.log(2.0)
        return 1.10 * ensemble_log_probabilities
>>>>>>> REPLACE