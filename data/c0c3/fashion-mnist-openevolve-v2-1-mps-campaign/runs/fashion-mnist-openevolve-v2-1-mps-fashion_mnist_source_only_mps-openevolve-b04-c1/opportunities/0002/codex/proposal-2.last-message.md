MECHANISM: Evaluation-time horizontal-flip logit ensemble

HYPOTHESIS: Averaging predictions for each validation image and its horizontal mirror will exceed 9,194 correct predictions because training already exposes the model to both orientations.

INTENDED_EDIT: Refactor the single-view forward pass and ensemble original and flipped views only during evaluation, without changing training or parameter count.

EVIDENCE: The current flip-augmented residual CNN achieves 9,194/10,000; its learned flip invariance makes a matched two-view ensemble a low-risk way to reduce prediction variance.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        return self.classifier(features)
=======
    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE