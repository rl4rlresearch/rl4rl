MECHANISM: Smoothed BatchNorm evaluation statistics

HYPOTHESIS: Reducing BatchNorm running-stat momentum from 0.10 to 0.05 will exceed 9,322 correct predictions by making validation normalization less sensitive to noisy late batches, including the undersized final batch.

INTENDED_EDIT: Preserve the best verified top-four architecture and training objective while applying a lower running-stat momentum to every BatchNorm layer.

EVIDENCE: Top-four attention with plain paired-view cross-entropy achieved the best result at 9,322 correct, while attention and loss modifications generally regressed; refining evaluation statistics is a parameter-neutral, computationally negligible orthogonal change.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.05
>>>>>>> REPLACE