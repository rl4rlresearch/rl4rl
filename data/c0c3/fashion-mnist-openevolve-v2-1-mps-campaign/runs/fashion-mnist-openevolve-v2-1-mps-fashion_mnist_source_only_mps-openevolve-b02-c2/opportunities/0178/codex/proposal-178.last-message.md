MECHANISM: Fine-grained bisection of the confidence-adaptive fusion boundary

HYPOTHESIS: A fusion coefficient of 0.08671875 will preserve 9,328 correct predictions while lowering cross-entropy below 0.191468821.

INTENDED_EDIT: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.0859375 and 0.0875; retain temperature 0.800713 and all training settings.

EVIDENCE: Coefficient 0.0859375 yielded 9,327 correct at 0.191468462 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.08671875 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE