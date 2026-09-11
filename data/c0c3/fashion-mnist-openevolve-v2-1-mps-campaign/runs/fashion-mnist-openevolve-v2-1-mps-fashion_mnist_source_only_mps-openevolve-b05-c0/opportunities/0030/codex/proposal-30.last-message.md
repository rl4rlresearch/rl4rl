MECHANISM: Moderately smoothed BatchNorm evaluation statistics

HYPOTHESIS: Setting BatchNorm momentum to 0.01 will exceed 9,254 correct predictions by reducing batch-to-batch noise in evaluation statistics while remaining responsive to the late training trajectory represented by the parameter EMA.

INTENDED_EDIT: Change all three BatchNorm layers from the default 0.1 momentum to 0.01 without altering model parameters, training compute, or the proven optimizer and EMA procedure.

EVIDENCE: BatchNorm previously increased correct predictions from 9,170 to 9,210, and the current parameter EMA raised the batch-32 result from 9,250 to 9,254; this motivates stabilizing the running statistics used with those averaged parameters, while choosing a more responsive horizon than the unverified 0.0025 attempt.

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
            nn.BatchNorm2d(24, momentum=0.01),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48, momentum=0.01),
            nn.GELU(),
>>>>>>> REPLACE