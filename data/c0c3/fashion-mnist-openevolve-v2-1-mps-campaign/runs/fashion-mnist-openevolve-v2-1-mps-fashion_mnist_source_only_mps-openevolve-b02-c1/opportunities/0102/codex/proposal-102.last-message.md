MECHANISM: Full-field global-logit shortcut

HYPOTHESIS: Adding a direct classifier over all 7×7 terminal features while preserving the proven 3×3 positional path will achieve at least 9,311 validation-correct predictions within the parameter and time limits.

INTENDED_EDIT: Move the unchanged terminal max-pool outside the feature extractor, reduce the dense head by one unit, and add its logits to a bias-free global-average classifier covering the complete pre-pool feature map.

EVIDENCE: The normalized 147-unit head improved the best result to 9,310 correct, whereas replacing its terminal pooling regressed to 9,234; an additive global shortcut preserves that successful path while cheaply recovering evidence from the row and column omitted by its 7×7-to-3×3 pool.

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.terminal_pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 146),
            nn.BatchNorm1d(146),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(146, 10),
        )
        self.global_classifier = nn.Linear(96, 10, bias=False)

    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        positional_logits = self.classifier(self.terminal_pool(features))
        global_logits = self.global_classifier(features.mean(dim=(2, 3)))
        return positional_logits + global_logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
=======
        for view in views[1:]:
            view_logits = self._classify(view)
            ensemble = torch.logaddexp(
>>>>>>> REPLACE

<<<<<<< SEARCH
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
=======
        for view_index, view in enumerate(views):
            flipped_logits = self._classify(view.flip(-1))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
>>>>>>> REPLACE