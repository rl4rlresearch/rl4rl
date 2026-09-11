MECHANISM: Parameter-free intra-stage residual feature reuse

HYPOTHESIS: Adding residual connections between equal-width convolutional layers will exceed 9,290 correct predictions by improving feature preservation and gradient flow during the limited 1,564-step training run without changing capacity or validated widths.

INTENDED_EDIT: Add residual additions within both 32-channel and 64-channel convolutional stages while preserving all parameters, optimization, augmentation, and evaluation behavior.

EVIDENCE: Reducing convolutional width to enlarge the head regressed from 9,290 to 9,264 correct, while changing AdamW adaptation regressed to 9,269; this motivates retaining the best architecture and optimizer while improving its information flow without additional parameters.

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features

        x = features[2](features[1](features[0](images)))
        residual = x
        x = features[5](features[4](features[3](x)))
        x = x + residual
        x = features[6](x)

        x = features[9](features[8](features[7](x)))
        residual = x
        x = features[12](features[11](features[10](x)))
        x = x + residual
        x = features[13](x)

        return self.classifier(x)
>>>>>>> REPLACE