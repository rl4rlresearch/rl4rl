MECHANISM: Cardinal-symmetric near-shift logit TTA

HYPOTHESIS: Independent 90%-max pooling with equal 0.09 weights for horizontal and vertical one-pixel TTA shifts will exceed 9,284 correct predictions by removing unsupported directional bias while retaining the distance-sensitive radius-two weighting absent from the 9,283-correct fully matched TTA design.

INTENDED_EDIT: Restore the best verified independent pooling gates and symmetrize only radius-one cardinal TTA weights, leaving the original distance-weighted outer shifts unchanged.

EVIDENCE: Independent pooling with the original TTA achieved 9,284 correct, while simultaneously symmetrizing cardinal shifts and flattening all radius-two weights achieved 9,283; isolating cardinal symmetry tests whether it is beneficial without discarding useful distance-based reliability weighting.

<<<<<<< SEARCH
        shared_pool = MixedPool2d()
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            shared_pool,
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            shared_pool,
=======
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
>>>>>>> REPLACE

<<<<<<< SEARCH
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
=======
                if radius == 1:
                    if delta_x == 0 or delta_y == 0:
                        weight = 0.09
                    else:
                        weight = 0.045
>>>>>>> REPLACE