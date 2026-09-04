MECHANISM: Validation-batch dense-feature normalization

HYPOTHESIS: BatchNorm1d using live validation-batch statistics will exceed 9,288 correct predictions by preserving the successful training-time normalization while eliminating the evaluation-statistics sensitivity demonstrated by momentum 0.1 versus 0.00623359375.

INTENDED_EDIT: Add BatchNorm1d after the first classifier linear layer with running statistics disabled, and restore the best-verified evaluation logit scale of 1.20514.

EVIDENCE: Dense BatchNorm with momentum 0.1 achieved the best result at 9,288 correct, while changing only its running-statistics momentum to 0.00623359375 produced 9,284 correct; this directly motivates evaluating with current-batch statistics instead of another moving-average rate.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = 1.17817 * logits
        return logits
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = 1.20514 * logits
        return logits
>>>>>>> REPLACE