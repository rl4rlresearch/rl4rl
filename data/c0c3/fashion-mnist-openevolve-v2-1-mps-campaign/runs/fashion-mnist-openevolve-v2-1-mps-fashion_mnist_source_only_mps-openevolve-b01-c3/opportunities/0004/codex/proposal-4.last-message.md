MECHANISM: Flip-ensemble inference

HYPOTHESIS: Averaging original and horizontally flipped logits at evaluation will exceed 9,086 correct predictions because the model is already trained with 50% horizontal flips, making both views class-preserving ensemble members.

INTENDED_EDIT: Keep training unchanged, but during evaluation average logits from each validation image and its horizontal reflection.

EVIDENCE: The current 9,086-correct design already applies random horizontal flips throughout training; flip-averaged inference directly exploits that learned invariance without adding parameters or reducing the fixed training exposure.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mid(self.stem(images))
        features = F.gelu(features + self.late2(self.late1(features)))
        return self.classifier(features)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.mid(self.stem(images))
        features = F.gelu(features + self.late2(self.late1(features)))
        logits = self.classifier(features)
        if self.training:
            return logits

        flipped = images.flip(-1)
        flipped_features = self.mid(self.stem(flipped))
        flipped_features = F.gelu(
            flipped_features + self.late2(self.late1(flipped_features))
        )
        flipped_logits = self.classifier(flipped_features)
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE