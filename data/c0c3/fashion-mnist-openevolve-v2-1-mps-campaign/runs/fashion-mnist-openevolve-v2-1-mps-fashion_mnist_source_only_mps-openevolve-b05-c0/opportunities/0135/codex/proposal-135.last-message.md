MECHANISM: Horizontal-reflection augmentation with test-time logit ensembling

HYPOTHESIS: Training on random horizontal reflections and averaging original/reflected evaluation logits will exceed 9,260 correct predictions by reducing orientation-specific prediction variance.

INTENDED_EDIT: Randomly reflect half of each training batch and average logits from original and reflected images during evaluation, retaining the calibrated positive logit multiplier.

EVIDENCE: EMA and BatchNorm-buffer refinements around the 9,260-correct design reduced accuracy, while temperature refinement only improved cross-entropy without changing predictions; an input-space invariance ensemble is therefore an orthogonal attempt to improve the accuracy-dominant objective.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.4164
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            reflected_logits = self.classifier(
                self.features(torch.flip(images, dims=(-1,)))
            )
            logits = 0.5 * (logits + reflected_logits)
            logits = logits * 1.4164
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return images, labels
=======
    del step, total_steps
    reflect = torch.rand(
        images.shape[0], device=images.device
    ) < 0.5
    images = torch.where(
        reflect[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
>>>>>>> REPLACE