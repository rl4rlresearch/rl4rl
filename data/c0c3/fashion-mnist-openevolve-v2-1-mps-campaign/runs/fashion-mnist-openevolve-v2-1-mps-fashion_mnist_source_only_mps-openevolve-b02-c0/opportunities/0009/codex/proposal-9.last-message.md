MECHANISM: Translation-and-flip probability ensemble

HYPOTHESIS: Averaging predictions over the centered image and four one-pixel translations will exceed 9,146 correct by exploiting the translation invariance learned from random-crop augmentation without changing the proven training configuration.

INTENDED_EDIT: Extend evaluation-time augmentation from two horizontal-flip views to ten views spanning center, one-pixel cardinal translations, and their flips.

EVIDENCE: The batch-48 architecture and training recipe remains best at 9,146 correct, while changing capacity, update density, and parameter averaging all regressed; retaining its learned solution and extending its already-successful flip averaging is the most targeted next test.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits
        flipped_logits = self._forward_once(images.flip(-1))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        height, width = images.shape[-2:]
        views = (
            images,
            padded[..., :height, 1 : width + 1],
            padded[..., 2 : height + 2, 1 : width + 1],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
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
>>>>>>> REPLACE