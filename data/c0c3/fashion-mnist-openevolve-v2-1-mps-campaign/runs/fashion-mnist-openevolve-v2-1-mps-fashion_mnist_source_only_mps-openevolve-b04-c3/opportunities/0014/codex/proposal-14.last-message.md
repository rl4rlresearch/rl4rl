MECHANISM: Probability-mixture flip ensemble

HYPOTHESIS: Restoring flip-only training and averaging flip predictions in probability space will exceed 9,229 correct predictions by preserving the qualified training procedure while producing a better-calibrated inference ensemble.

INTENDED_EDIT: Remove the harmful random translations and replace logit averaging at validation with an arithmetic mixture of class probabilities represented as log-probabilities.

EVIDENCE: Flip-only batch-64 training achieved 9,229 correct, whereas adding translations reduced correctness to 9,058; probability-space fusion improves the existing flip ensemble without changing training cost or parameters.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=-1),
                F.log_softmax(flipped_logits, dim=-1),
            ) - math.log(2.0)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    images = F.pad(images, (2, 2, 2, 2))
    top, left = torch.randint(0, 5, (2,)).tolist()
    images = images[..., top : top + 28, left : left + 28]
    return images, labels
=======
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
>>>>>>> REPLACE