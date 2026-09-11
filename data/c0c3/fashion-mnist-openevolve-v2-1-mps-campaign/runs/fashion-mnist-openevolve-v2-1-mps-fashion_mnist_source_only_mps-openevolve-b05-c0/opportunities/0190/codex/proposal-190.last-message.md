MECHANISM: Confidence-gated translation ensembling

HYPOTHESIS: Applying four-direction one-pixel translation ensembling only to predictions with an uncalibrated top-two logit margin below 0.75 will increase validation_correct above 9,260 while completing within the time limit.

INTENDED_EDIT: During evaluation, retain original logits for confident images and blend original logits equally with the four-shift mean for uncertain images before applying the incumbent calibration.

EVIDENCE: Full four-direction translation ensembling timed out, while horizontal reflection reduced accuracy because it changes class-relevant orientation; confidence gating preserves the class-safe translation idea while substantially reducing its evaluation cost and limiting prediction changes to plausible boundary errors.

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
            uncertain = (top_two[:, 0] - top_two[:, 1]) < 0.75
            if uncertain.any():
                selected = images[uncertain]
                padded = F.pad(selected, (1, 1, 1, 1), mode="replicate")
                shifted = torch.cat(
                    (
                        padded[:, :, 0:28, 1:29],
                        padded[:, :, 2:30, 1:29],
                        padded[:, :, 1:29, 0:28],
                        padded[:, :, 1:29, 2:30],
                    ),
                    dim=0,
                )
                shifted_logits = self.classifier(self.features(shifted))
                shifted_logits = shifted_logits.reshape(
                    4, selected.shape[0], 10
                ).mean(dim=0)
                logits = logits.clone()
                logits[uncertain] = (
                    0.5 * logits[uncertain] + 0.5 * shifted_logits
                )
            logits = logits * 1.416375
        return logits
>>>>>>> REPLACE