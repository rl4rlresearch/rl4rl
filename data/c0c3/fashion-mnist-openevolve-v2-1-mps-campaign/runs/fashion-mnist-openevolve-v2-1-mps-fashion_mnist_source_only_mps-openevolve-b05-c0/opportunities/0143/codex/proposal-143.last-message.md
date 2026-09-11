MECHANISM: Mild dense-head dropout regularization

HYPOTHESIS: Applying 10% dropout to the parameter-dominant classifier hidden layer will increase validation_correct above 9,260 by reducing co-adaptation without imposing harmful image-space invariances.

INTENDED_EDIT: Add training-only dropout after the classifier’s hidden GELU while preserving the architecture size, optimizer, EMA, label smoothing, and evaluation calibration.

EVIDENCE: Both label-smoothing directions reduced correctness, translation and reflection augmentation were harmful, and the projected residual block reached only 9,246; mild parameter-free regularization of the large dense head is an orthogonal, localized test.

<<<<<<< SEARCH
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
=======
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(88, 10),
>>>>>>> REPLACE