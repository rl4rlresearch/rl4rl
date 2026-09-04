MECHANISM: Evaluation-only temperature sharpening

HYPOTHESIS: Sharpening evaluation logits by a temperature of 0.95 will preserve all 9,319 argmax-correct predictions while reducing validation cross-entropy below 0.2246338.

INTENDED_EDIT: Divide logits by 0.95 only during evaluation, leaving training, parameters, and optimizer behavior unchanged.

EVIDENCE: The prior temperature-scaling verification timed out rather than producing negative accuracy evidence; label smoothing can leave logits mildly underconfident, and positive temperature scaling mathematically preserves every argmax prediction.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits / 0.95
        return logits
>>>>>>> REPLACE