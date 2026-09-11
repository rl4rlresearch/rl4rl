MECHANISM: Stronger distance-stratified outer-shift ensembling

HYPOTHESIS: Restoring the verified 40-local/24-context model and reallocating radius-2 corner weight to axial shifts will exceed 9,268 correct predictions while preserving total ensemble weight.

INTENDED_EDIT: Adopt the best multiscale aggregator and strengthen its successful TTA distance bias: axial, knight, and corner radius-2 shifts receive 0.0125, 0.00625, and 0 respectively.

EVIDENCE: Distance-stratified radius-2 TTA preserved 9,268 correct and reduced cross-entropy from 0.2124652 to 0.2124557; this controlled dose increase further favors the least-displaced outer views.

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
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
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
=======
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.0125
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.0
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE