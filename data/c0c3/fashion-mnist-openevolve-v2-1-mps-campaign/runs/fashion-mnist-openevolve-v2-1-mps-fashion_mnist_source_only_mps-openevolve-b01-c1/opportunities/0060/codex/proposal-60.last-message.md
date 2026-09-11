MECHANISM: Residual spatial-pyramid classifier

HYPOTHESIS: Replacing the shallow fixed-coordinate decoder with bottleneck spatial refinement and coarse multi-scale pooling will exceed 9,247 correct predictions by learning part-to-whole structure with smoother translation behavior.

INTENDED_EDIT: Add two efficient residual bottleneck blocks and replace the flattened 7×7 head with a 1×1/2×2/3×3 spatial-pyramid descriptor and wider classifier, totaling 248,402 learned parameters.

EVIDENCE: Widening the flattened head regressed to 9,210 and appended global summaries reached only 9,236, indicating that capacity added within the fixed-coordinate mechanism was insufficient; this patch instead reallocates capacity to spatial feature computation and structured aggregation while avoiding the timed-out attention-pooling mechanism.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
        self.refinement = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(48, 24, kernel_size=1, bias=False),
                    nn.BatchNorm2d(24),
                    nn.GELU(),
                    nn.Conv2d(
                        24,
                        24,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(24),
                    nn.GELU(),
                    nn.Conv2d(24, 48, kernel_size=1, bias=False),
                    nn.BatchNorm2d(48),
                )
                for _ in range(2)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(48 * (1 + 4 + 9), 280),
            nn.GELU(),
            nn.LayerNorm(280),
            nn.Linear(280, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        for block in self.refinement:
            features = F.gelu(features + block(features))
        pyramid = torch.cat(
            [
                F.adaptive_avg_pool2d(features, size).flatten(1)
                for size in (1, 2, 3)
            ],
            dim=1,
        )
        return self.classifier(pyramid)
>>>>>>> REPLACE