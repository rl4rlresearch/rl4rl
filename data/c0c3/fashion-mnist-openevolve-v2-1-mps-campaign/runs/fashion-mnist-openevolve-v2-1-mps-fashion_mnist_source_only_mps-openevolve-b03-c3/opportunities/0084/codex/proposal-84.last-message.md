MECHANISM: Higher-update small-minibatch optimization

HYPOTHESIS: Restoring the verified 40-local/24-context model while reducing the batch size to 128 will exceed 9,268 correct predictions because the fixed 100,000-example exposure will provide roughly 782 optimizer steps instead of 522.

INTENDED_EDIT: Restore the best multiscale architecture, mildly distance-stratified TTA, and 1.253 calibration, then reduce only the training batch size from 192 to 128.

EVIDENCE: The 40-local/24-context design reached 9,268 correct, while subsequent architecture, TTA-aggregation, augmentation-alignment, and EMA changes did not improve it; every reported design used batch size 192, leaving optimizer-update density as an untested lever.

<<<<<<< SEARCH
BATCH_SIZE = 192
=======
BATCH_SIZE = 128
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
=======
        self.aggregate_local = nn.Sequential(
            nn.Conv2d(
                80, 40, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(40),
            nn.GELU(),
        )
        self.aggregate_context = nn.Sequential(
            nn.Conv2d(
                80,
                24,
                kernel_size=3,
                stride=2,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 140),
            nn.LayerNorm(140),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(140, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
            (
                self.aggregate_local(features),
                self.aggregate_context(features),
            ),
            dim=1,
        )
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
        return 1.285 * logits
=======
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
        return 1.253 * logits
>>>>>>> REPLACE