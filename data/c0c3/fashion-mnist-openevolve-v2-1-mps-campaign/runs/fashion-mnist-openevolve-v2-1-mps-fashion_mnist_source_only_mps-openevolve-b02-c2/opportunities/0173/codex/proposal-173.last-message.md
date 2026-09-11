MECHANISM: Boundary-seeking confidence-adaptive flip fusion

HYPOTHESIS: A fusion-margin coefficient of 0.075 will retain 9,328 correct predictions while reducing validation cross-entropy below 0.191472037.

INTENDED_EDIT: Reduce the evaluation-only flip-fusion margin coefficient from 0.10 to 0.075; leave training and temperature calibration unchanged.

EVIDENCE: Coefficient 0.05 achieved lower cross-entropy (0.191462738) but 9,327 correct, while 0.10 achieved 9,328 correct at 0.191472037; their midpoint tests whether the additional correct prediction can be retained with better calibration.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.075 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE