MECHANISM: Conservative one-pixel translation logit ensemble

HYPOTHESIS: Original-weighted averaging with four one-pixel translated views will exceed 9,260 correct predictions by reducing dense-head sensitivity to minor image alignment while preserving the original prediction as half of the ensemble.

INTENDED_EDIT: During evaluation, combine original-view logits at 50% weight with replicate-padded one-pixel left, right, up, and down translations, then apply the verified calibration multiplier; training remains unchanged.

EVIDENCE: Horizontal-reflection ensembling reduced correctness from 9,260 to 8,876, showing that transformation semantics are critical; one-pixel translations are class-preserving and test spatial robustness without perturbing the verified training trajectory.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.4164
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.classifier(self.features(images))

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = torch.cat(
            (
                images,
                padded[:, :, 1:-1, :-2],
                padded[:, :, 1:-1, 2:],
                padded[:, :, :-2, 1:-1],
                padded[:, :, 2:, 1:-1],
            ),
            dim=0,
        )
        view_logits = self.classifier(self.features(views)).reshape(
            5, images.shape[0], 10
        )
        logits = (
            4.0 * view_logits[0] + view_logits[1:].sum(dim=0)
        ) / 8.0
        return logits * 1.4164
>>>>>>> REPLACE