MECHANISM: Distance-stratified outer-shift logit ensembling

HYPOTHESIS: Restoring the verified 40-local/24-context architecture and redistributing the unchanged 0.10 radius-2 TTA weight toward less-displaced axial views will exceed 9,268 correct predictions.

INTENDED_EDIT: Adopt the best verified multiscale aggregator, then weight radius-2 axial, knight, and corner shifts by 0.009375, 0.00625, and 0.003125 respectively while preserving total ensemble weight and calibration.

EVIDENCE: The 40/24 aggregator achieved the best result at 9,268 correct, while further architectural changes failed. Radius-1 evaluation already downweights diagonal shifts, but radius-2 currently assigns equal weight despite different displacement magnitudes, motivating this controlled evaluation-only refinement.

<<<<<<< SEARCH
        self.aggregate = nn.Sequential(
            nn.Conv2d(
                80, 64, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = self.aggregate(features)
        return self.classifier(features)
=======
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
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE