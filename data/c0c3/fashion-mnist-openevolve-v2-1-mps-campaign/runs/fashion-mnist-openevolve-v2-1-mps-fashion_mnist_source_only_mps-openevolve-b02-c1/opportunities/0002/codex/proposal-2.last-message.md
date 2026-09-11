MECHANISM: Flip-ensemble test-time augmentation

HYPOTHESIS: Averaging class probabilities for original and horizontally flipped validation images will exceed 9,097 correct predictions because training already enforces flip invariance.

INTENDED_EDIT: Keep training unchanged, but ensemble original and mirrored predictions during evaluation.

EVIDENCE: The 9,097-correct design trains with random horizontal flips, so paired evaluation can reduce residual orientation-dependent prediction variance without adding parameters or changing exposure.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)
>>>>>>> REPLACE