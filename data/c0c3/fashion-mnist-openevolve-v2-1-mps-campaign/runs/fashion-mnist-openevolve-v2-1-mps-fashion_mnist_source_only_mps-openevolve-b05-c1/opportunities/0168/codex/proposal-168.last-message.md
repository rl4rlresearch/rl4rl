MECHANISM: Reduced bottleneck dropout under fixed exposure

HYPOTHESIS: Halving classifier dropout from 0.10 to 0.05 will exceed 9,360 correct predictions by improving optimization of the narrow 30-unit head while retaining mild regularization.

INTENDED_EDIT: Reduce only the classifier-head dropout probability; preserve architecture, parameter count, optimizer, EMA, augmentation, TTA, and calibration.

EVIDENCE: Added regularization through label smoothing fell to 9,332 correct and focal reweighting fell to 9,331, while the uniform-loss baseline reached 9,360; this motivates cautiously reducing existing training noise.

<<<<<<< SEARCH
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
=======
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(30, 10),
>>>>>>> REPLACE