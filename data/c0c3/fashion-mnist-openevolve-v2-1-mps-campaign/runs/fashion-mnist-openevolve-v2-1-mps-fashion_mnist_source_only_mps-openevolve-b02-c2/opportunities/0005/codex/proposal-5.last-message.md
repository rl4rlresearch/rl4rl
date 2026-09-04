MECHANISM: Low-resolution bottleneck residual refinement

HYPOTHESIS: Extending the 9,214-correct Reference Design 3 with a 32-channel bottleneck residual block at 7×7 resolution will exceed 9,214 correct predictions while retaining its successful batch-128 training, classifier capacity, augmentation, and flip ensemble.

INTENDED_EDIT: Adopt Reference Design 3’s high-resolution residual block and evaluation ensemble, then add a 64→32→32→64 bottleneck residual block before the classifier, bringing the model to approximately 241,274 learned parameters.

EVIDENCE: Reference Design 3 improved from 9,161 to 9,214 correct by adding residual spatial refinement while preserving the proven regimen; the bottleneck adds further spatial processing within the parameter ceiling without the classifier shrinkage, batch-size change, or translation augmentation that confounded Reference Design 2.

<<<<<<< SEARCH
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(32)

        self.block_conv1 = nn.Conv2d(
=======
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(32)
        self.stem_refine_conv1 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm1 = nn.BatchNorm2d(32)
        self.stem_refine_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm2 = nn.BatchNorm2d(32)

        self.block_conv1 = nn.Conv2d(
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pool = nn.MaxPool2d(2)

        self.classifier = nn.Sequential(
=======
        self.pool = nn.MaxPool2d(2)

        self.deep_conv1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.deep_norm1 = nn.BatchNorm2d(32)
        self.deep_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.deep_norm2 = nn.BatchNorm2d(32)
        self.deep_conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.deep_norm3 = nn.BatchNorm2d(64)

        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(F.gelu(self.stem_norm(self.stem_conv(images))))
        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.stem_norm(self.stem_conv(images)))
        residual = features
        features = F.gelu(
            self.stem_refine_norm1(self.stem_refine_conv1(features))
        )
        features = self.stem_refine_norm2(self.stem_refine_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = features
        features = F.gelu(self.deep_norm1(self.deep_conv1(features)))
        features = F.gelu(self.deep_norm2(self.deep_conv2(features)))
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
>>>>>>> REPLACE