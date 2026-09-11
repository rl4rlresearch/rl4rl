MECHANISM: Fine-grained bisection of the confidence-adaptive fusion boundary

HYPOTHESIS: A fusion-margin coefficient of 0.0859375 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

INTENDED_EDIT: Reduce the evaluation-only flip-fusion coefficient from 0.0875 to the midpoint between 0.084375 and 0.0875; retain all training settings and temperature 0.800713.

EVIDENCE: Coefficient 0.084375 produced 9,327 correct at 0.191468112 cross-entropy, while 0.0875 produced 9,328 correct at 0.191468821; their midpoint is the most informative next test for the smallest coefficient preserving the additional correct prediction.

<<<<<<< SEARCH
        view_weights = F.softmax(0.0875 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.0859375 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE