MECHANISM: Identity-initialized global channel recalibration

HYPOTHESIS: Adding a lightweight input-conditioned channel gate to the verified 40-local/24-context model will exceed 9,268 correct predictions by adapting its feature mixture to each garment while preserving the successful spatial classifier and starting from an exact identity transformation.

INTENDED_EDIT: Restore the best mildly distance-stratified radius-2 TTA and add an identity-initialized squeeze-style gate before the multiscale aggregators, raising the parameter count from 245,150 to 249,118.

EVIDENCE: The 40-local/24-context flattened model achieved the best 9,268 correct predictions, whereas channel reallocation and replacement pooling heads regressed; this motivates retaining that topology while using the remaining parameter budget for conditional channel recalibration.

<<<<<<< SEARCH
        self.aggregate_local = nn.Sequential(
=======
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(80, 24, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(24, 80, kernel_size=1),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        self.aggregate_local = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
=======
        features = F.gelu(features + self.refine(features))
        features = features * (
            1.0 + torch.tanh(self.channel_gate(features))
        )
        features = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                    if manhattan_distance == 2:
                        weight = 0.0125
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.0
=======
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
>>>>>>> REPLACE