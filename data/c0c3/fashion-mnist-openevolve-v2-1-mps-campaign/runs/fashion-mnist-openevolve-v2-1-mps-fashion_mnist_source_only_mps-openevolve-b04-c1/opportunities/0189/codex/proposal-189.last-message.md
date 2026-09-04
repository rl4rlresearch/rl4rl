MECHANISM: Wider position-sensitive classifier projection

HYPOTHESIS: Increasing the spatial classifier width from 48 to 56 units will exceed 9,330 correct predictions by reducing the post-fusion representation bottleneck while remaining below the 250,000-parameter ceiling.

INTENDED_EDIT: Widen both classifier linear layers consistently from 48 to 56 hidden units, raising the parameter count from 224,442 to 249,618 without changing the proven training procedure.

EVIDENCE: Global pooling reduced correctness to 9,290 and decision-level flip symmetrization reached only 9,285, showing that preserving and effectively projecting the full 7×7 spatial representation matters; this change adds capacity directly to that proven position-sensitive path.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(56, 10),
        )
>>>>>>> REPLACE