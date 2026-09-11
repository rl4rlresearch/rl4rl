MECHANISM: Exposure-aligned BatchNorm running statistics

HYPOTHESIS: Setting BatchNorm momentum to 0.0025 will exceed 9,254 correct predictions by aligning its roughly 400-step evaluation-statistics horizon with the strided tail EMA’s effective parameter horizon.

INTENDED_EDIT: Change only the three BatchNorm running-statistics momentum values from the default 0.1 to 0.0025.

EVIDENCE: BatchNorm improved correct predictions from 9,170 to 9,210, and exposure-normalized strided EMA improved the batch-32 result from 9,250 to 9,254; matching BatchNorm’s statistics horizon to that EMA is a runtime-neutral follow-up.

<<<<<<< SEARCH
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
=======
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24, momentum=0.0025),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48, momentum=0.0025),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48, momentum=0.0025),
            nn.GELU(),
>>>>>>> REPLACE