MECHANISM: Intermediate confidence-adaptive flip fusion

HYPOTHESIS: A 0.15 fusion-margin coefficient will avoid the harmful over-weighting observed at 0.20 while potentially correcting disagreement cases not changed at 0.10, exceeding 9,328 correct predictions.

INTENDED_EDIT: Increase the evaluation-only flip-fusion margin coefficient from 0.10 to the midpoint 0.15; retain constant 0.02 smoothing and temperature 0.800713.

EVIDENCE: Coefficient 0.10 achieved 9,328 correct with 0.191472037 cross-entropy, while 0.20 changed the decision boundary and fell to 9,327; testing the midpoint isolates whether useful decisive-view corrections occur before the harmful flip.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.15 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE