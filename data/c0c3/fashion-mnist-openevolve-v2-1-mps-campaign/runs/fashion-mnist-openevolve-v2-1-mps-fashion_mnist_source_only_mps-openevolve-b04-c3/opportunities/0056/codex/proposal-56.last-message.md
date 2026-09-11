MECHANISM: 0.625-order probability power-mean flip ensemble

HYPOTHESIS: A 0.625-order power mean will retain 9,242 correct predictions while reducing cross-entropy below the verified 0.75-order ensemble’s 0.209166.

INTENDED_EDIT: Remove the regressive global-average shortcut, restore the verified 245,040-parameter classifier, and ensemble original and flipped predictions with a 0.625-order probability mean.

EVIDENCE: Orders 0.75, 1.0, and 1.25 all achieved 9,242 correct while cross-entropy improved as order decreased; geometric averaging had still lower cross-entropy but 9,240 correct, motivating a conservative interpolation below 0.75 after the 0.5-order attempt timed out.

<<<<<<< SEARCH
        self.global_classifier = nn.Linear(96, 10)
        nn.init.zeros_(self.global_classifier.weight)
        nn.init.zeros_(self.global_classifier.bias)

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        global_features = features.mean(dim=(-2, -1))
        return self.classifier(features) + self.global_classifier(global_features)
=======
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
        if not self.training:
            flipped_logits = self._classify(images.flip(-1))
            power = 0.625
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
>>>>>>> REPLACE