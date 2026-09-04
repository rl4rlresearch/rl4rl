MECHANISM: Capacity-neutral residual channel recalibration

HYPOTHESIS: Adding lightweight channel attention while restoring the verified 5% warmup and 1.15 evaluation scale will exceed 9,330 correct predictions by adaptively emphasizing useful residual features without increasing computation enough to risk the time limit.

INTENDED_EDIT: Add a squeeze-and-excitation gate to the 7×7 residual branch, reduce the classifier width from 35 to 34 to remain below 250,000 parameters, and restore the best verified schedule and calibration.

EVIDENCE: The 5% warmup and 1.15-scaled symmetric ensemble achieved 9,330 correct, while changes to fusion, EMA, and target scheduling were worse; this motivates preserving the proven training procedure and testing a parameter-efficient architectural improvement.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 35),
            nn.LayerNorm(35),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(35, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
        self.refine_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 34),
            nn.LayerNorm(34),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(34, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        refined = self.refine(features)
        refined = refined * (2.0 * self.refine_gate(refined))
        features = F.gelu(features + refined)
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE

<<<<<<< SEARCH
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE