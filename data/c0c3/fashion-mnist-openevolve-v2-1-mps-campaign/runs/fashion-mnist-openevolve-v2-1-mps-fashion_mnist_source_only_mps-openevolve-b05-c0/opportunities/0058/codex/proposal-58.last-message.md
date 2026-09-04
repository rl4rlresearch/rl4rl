MECHANISM: Two-view reflection logit ensemble

HYPOTHESIS: Averaging original and horizontally reflected evaluation logits will exceed 9,258 correct predictions by reducing spatial-orientation sensitivity without adding training work.

INTENDED_EDIT: Keep training unchanged, but evaluate each image together with its horizontal reflection and average their logits before applying the existing calibration.

EVIDENCE: The current EMA design reached 9,258 correct, while the translation-augmentation attempt identified spatial invariance as promising but timed out; evaluation-only ensembling tests that mechanism without changing the 100,000-example training path.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.1
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.classifier(self.features(images))

        batch_size = images.shape[0]
        views = torch.cat((images, torch.flip(images, dims=(-1,))), dim=0)
        view_logits = self.classifier(self.features(views))
        logits = 0.5 * (
            view_logits[:batch_size] + view_logits[batch_size:]
        )
        return logits * 1.1
>>>>>>> REPLACE