MECHANISM: Shared local concept bank with class-specific spatial readout

HYPOTHESIS: Replacing the 48-unit global dense bottleneck with 288 position-shared nonlinear concepts and a direct position-sensitive class readout will exceed 9,330 correct predictions by allocating capacity to reusable local evidence while preserving the valuable 7×7 layout.

INTENDED_EDIT: Replace the flatten-first classifier with a pointwise 64→288 concept projection, normalization, channel dropout, and direct linear readout over all concept-position pairs; the model remains below 250,000 learned parameters.

EVIDENCE: Widening the global dense projection fell to 9,301 and global pooling fell to 9,290, while local residual refinement reached only 9,309. This challenges the shared assumption that features should be globally mixed before their main nonlinearity: the new head instead applies a wide nonlinear basis locally, then aggregates class-specific spatial evidence.

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
            nn.Conv2d(64, 288, kernel_size=1, bias=False),
            nn.BatchNorm2d(288),
            nn.GELU(),
            nn.Dropout2d(0.15),
            nn.Flatten(),
            nn.Linear(288 * 7 * 7, 10),
        )
>>>>>>> REPLACE