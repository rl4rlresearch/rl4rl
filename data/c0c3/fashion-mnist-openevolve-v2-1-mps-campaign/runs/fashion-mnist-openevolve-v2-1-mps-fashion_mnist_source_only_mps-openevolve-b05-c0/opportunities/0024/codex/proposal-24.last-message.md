MECHANISM: Mild hidden-representation dropout

HYPOTHESIS: Applying 10% dropout to the classifier hidden layer will exceed 9,254 correct predictions by reducing co-adaptation in the parameter-dominant fully connected head.

INTENDED_EDIT: Insert dropout after the classifier’s hidden GELU activation while preserving all other architecture and training settings.

EVIDENCE: Label smoothing 0.05 outperformed both hard targets and 0.10 smoothing, showing that moderate regularization helps; hidden dropout tests an orthogonal, targeted form of moderate regularization.

<<<<<<< SEARCH
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
=======
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(88, 10),
>>>>>>> REPLACE