MECHANISM: Evaluation-time flip ensembling

HYPOTHESIS: Averaging original and horizontally flipped logits will exceed 8,994 correct predictions because the verified model was trained with 50% horizontal flips and already reached 89.94% accuracy.

INTENDED_EDIT: Preserve single-view training while using a two-view logit ensemble whenever the model is in evaluation mode.

EVIDENCE: The current 230,442-parameter design achieved 8,994 correct predictions with horizontal-flip augmentation, directly motivating inference-time averaging over the same label-preserving transformation.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)
        views = torch.cat((images, images.flip(-1)), dim=0)
        logits = self._forward_once(views)
        original_logits, flipped_logits = logits.chunk(2, dim=0)
        return 0.5 * (original_logits + flipped_logits)
>>>>>>> REPLACE