MECHANISM: Boundary-seeking confidence-adaptive flip fusion

HYPOTHESIS: A fusion-margin coefficient of 0.0875 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191472037.

INTENDED_EDIT: Reduce the evaluation-only flip-fusion margin coefficient from 0.10 to 0.0875, leaving training and temperature calibration unchanged.

EVIDENCE: Coefficient 0.075 produced 9,327 correct at 0.191466201 cross-entropy, while 0.10 produced 9,328 correct at 0.191472037; their midpoint efficiently searches for the lowest coefficient that preserves the additional correct prediction.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.0875 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE