MECHANISM: Intra-stage residual feature learning

HYPOTHESIS: Adding parameter-free residual paths across each same-width convolutional pair will raise validation_correct above 9,252 by improving optimization during the fixed two exposures.

INTENDED_EDIT: Replace the sequential feature forward with two residual convolutional stages while preserving all layers, parameters, augmentation, optimizer, and inference ensemble.

EVIDENCE: The current architecture reached 9,252 correct, while inference-only calibration changes cannot improve its argmax and subsequent loss changes yielded no completed accuracy evidence; residual paths directly target faster feature learning without increasing parameters or model passes.

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features[2](self.features[1](self.features[0](images)))
        residual = features
        features = self.features[4](self.features[3](features))
        features = self.features[5](features + residual)
        features = self.features[7](self.features[6](features))

        features = self.features[10](self.features[9](self.features[8](features)))
        residual = features
        features = self.features[12](self.features[11](features))
        features = self.features[13](features + residual)
        features = self.features[15](self.features[14](features))
        return self.classifier(features)
>>>>>>> REPLACE