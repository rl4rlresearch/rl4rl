MECHANISM: Evaluation-time horizontal-reflection ensembling

HYPOTHESIS: Averaging original-image and horizontally flipped logits will increase validation_correct above 9,260 by reducing orientation-sensitive errors on otherwise unchanged class semantics.

INTENDED_EDIT: During evaluation only, average logits from each image and its horizontal reflection before applying the incumbent calibration scale; training remains unchanged.

EVIDENCE: Fine calibration probes around 1.416375 and terminal-EMA refinements preserved 9,260 predictions but produced no strict improvement, motivating an orthogonal change capable of correcting argmax decisions without disturbing the verified training trajectory.

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
            flipped_logits = self.classifier(
                self.features(torch.flip(images, dims=(-1,)))
            )
            logits = 0.5 * (logits + flipped_logits)
            logits = logits * 1.416375
        return logits
>>>>>>> REPLACE