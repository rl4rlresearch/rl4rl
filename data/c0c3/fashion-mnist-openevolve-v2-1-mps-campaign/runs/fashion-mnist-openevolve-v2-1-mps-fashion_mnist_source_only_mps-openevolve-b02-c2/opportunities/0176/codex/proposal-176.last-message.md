MECHANISM: Fine-grained bisection of the confidence-adaptive fusion boundary

HYPOTHESIS: A fusion-margin coefficient of 0.084375 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

INTENDED_EDIT: Set the evaluation-only flip-fusion coefficient to the midpoint between the verified 0.08125 and 0.0875 designs; retain temperature 0.800713 and all training settings.

EVIDENCE: Coefficient 0.08125 produced 9,327 correct with 0.191467437 cross-entropy, while 0.0875 produced 9,328 correct with 0.191468821; bisection is the most informative search for the smallest coefficient preserving the additional correct prediction.

<<<<<<< SEARCH
        view_weights = F.softmax(0.05 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.084375 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE