MECHANISM: Conservative one-pixel translation ensemble

HYPOTHESIS: Evaluation-time ensembling with four class-preserving one-pixel translations, weighted equally with a 50% original-image contribution, will increase validation_correct above 9,260 without the semantic distortion caused by horizontal reflection.

INTENDED_EDIT: During evaluation only, combine the original logits with logits from one-pixel shifts in all four cardinal directions using replicated boundary pixels, then apply the incumbent calibration scale.

EVIDENCE: Horizontal-reflection ensembling reduced validation_correct from 9,260 to 8,877, showing that orientation-changing augmentation is harmful; small translations preserve class semantics and test whether spatial instability can correct argmax errors while leaving the verified training trajectory unchanged.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.416375
        return logits
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
            shifted_images = torch.cat(
                (
                    padded[:, :, 1:29, 0:28],
                    padded[:, :, 1:29, 2:30],
                    padded[:, :, 0:28, 1:29],
                    padded[:, :, 2:30, 1:29],
                ),
                dim=0,
            )
            shifted_logits = self.classifier(self.features(shifted_images))
            shifted_logits = shifted_logits.reshape(
                4, images.shape[0], 10
            ).sum(dim=0)
            logits = (4.0 * logits + shifted_logits) / 8.0
            logits = logits * 1.416375
        return logits
>>>>>>> REPLACE