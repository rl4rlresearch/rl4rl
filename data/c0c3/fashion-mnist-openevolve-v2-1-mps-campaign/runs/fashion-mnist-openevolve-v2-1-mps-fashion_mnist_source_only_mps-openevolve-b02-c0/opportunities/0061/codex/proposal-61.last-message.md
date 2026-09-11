MECHANISM: Single-view inference fast path

HYPOTHESIS: Eliminating test-time augmentation will finish within the verification limit while retaining at least 9,100 correct predictions, establishing whether evaluation overhead caused the repeated timeouts.

INTENDED_EDIT: Replace the ten-view, five-pass evaluation ensemble with one calibrated center-view forward pass; training remains unchanged.

EVIDENCE: The verified ten-view design reached 9,252 correct but took 76.9 training seconds, and every subsequent variant timed out—including the two-view center/flip ensemble—so halving that latest evaluation workload is the most direct completion-oriented test.

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
        return 1.10 * self._forward_once(images)
>>>>>>> REPLACE