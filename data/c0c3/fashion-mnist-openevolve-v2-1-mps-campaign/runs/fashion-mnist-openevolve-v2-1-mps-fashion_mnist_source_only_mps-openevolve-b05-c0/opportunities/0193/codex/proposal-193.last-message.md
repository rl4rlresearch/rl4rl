MECHANISM: Narrow-margin horizontal translation ensembling

HYPOTHESIS: Applying symmetric one-pixel translation ensembling only when the uncalibrated top-two margin is below 0.25 will increase validation_correct above 9,260 while remaining within the verification time limit.

INTENDED_EDIT: For low-margin evaluation examples, blend original logits with left/right replicated-boundary translations computed in one packed auxiliary forward; preserve incumbent calibration and training.

EVIDENCE: Full and 0.75-margin four-direction translation ensembles timed out, while translation training reduced accuracy; stricter evaluation gating and two translations retain the class-preserving test-time intervention with substantially less computation and exposure.

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
            top_two = logits.topk(2, dim=1).values
            uncertain = (top_two[:, 0] - top_two[:, 1]) < 0.25
            if uncertain.any():
                selected = images[uncertain]
                padded = F.pad(selected, (1, 1, 0, 0), mode="replicate")
                translated = torch.cat(
                    (padded[..., :-2], padded[..., 2:]), dim=0
                )
                translated_logits = self.classifier(
                    self.features(translated)
                )
                left_logits, right_logits = translated_logits.chunk(2, dim=0)
                blended = (
                    0.5 * logits[uncertain]
                    + 0.25 * left_logits
                    + 0.25 * right_logits
                )
                logits = logits.clone()
                logits[uncertain] = blended
            logits = logits * 1.416375
        return logits
>>>>>>> REPLACE