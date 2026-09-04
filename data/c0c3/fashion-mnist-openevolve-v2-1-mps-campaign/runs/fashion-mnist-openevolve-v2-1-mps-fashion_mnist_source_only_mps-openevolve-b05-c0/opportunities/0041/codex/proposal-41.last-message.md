MECHANISM: Mild classifier-head dropout

HYPOTHESIS: Adding 0.1 dropout to the parameter-dominant dense head will exceed 9,254 correct predictions by reducing hidden-feature co-adaptation without materially increasing runtime.

INTENDED_EDIT: Insert dropout after the classifier’s hidden GELU while retaining the proven architecture, optimizer, EMA, label smoothing, batch size, and evaluation scaling.

EVIDENCE: Label smoothing 0.05 outperformed both hard targets and 0.10 smoothing, showing mild regularization helps; the classifier contains 207,954 of 239,634 parameters, making its hidden layer the most targeted place for additional low-cost regularization.

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