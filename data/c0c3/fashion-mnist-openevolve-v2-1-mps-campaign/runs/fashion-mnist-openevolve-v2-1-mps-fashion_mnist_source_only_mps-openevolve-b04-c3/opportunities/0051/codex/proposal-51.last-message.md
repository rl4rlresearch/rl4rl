MECHANISM: Reduced classifier regularization under fixed two-pass exposure

HYPOTHESIS: Removing the classifier dropout while restoring the verified 12.5%-floor design will exceed 9,240 correct predictions by improving limited-exposure fitting without adding runtime or parameters.

INTENDED_EDIT: Remove the regressive global-average shortcut and disable the 10% classifier dropout, preserving the strongest verified backbone, optimizer, flip ensemble, and cosine schedule.

EVIDENCE: The unsmoothed reference design achieved 9,240 correct, while 2% label smoothing fell to 9,220 and the global shortcut fell to 9,230; both regressions support testing less regularization and restoring the proven single classifier.

<<<<<<< SEARCH
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
        self.global_classifier = nn.Linear(96, 10)
        nn.init.zeros_(self.global_classifier.weight)
        nn.init.zeros_(self.global_classifier.bias)

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        global_features = features.mean(dim=(-2, -1))
        return self.classifier(features) + self.global_classifier(global_features)
=======
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Linear(38, 10),
        )

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
>>>>>>> REPLACE