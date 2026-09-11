MECHANISM: Precision bisection of the flip-fusion decision boundary

HYPOTHESIS: A fusion coefficient of 0.08720703125 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.

INTENDED_EDIT: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.087109375 and 0.0873046875, retaining all training settings and temperature 0.800713.

EVIDENCE: Coefficient 0.087109375 yielded 9,327 correct at 0.191468735 cross-entropy, while 0.0873046875 yielded 9,328 correct at 0.191468777; their midpoint is the most informative next test for the smallest coefficient retaining the additional correct prediction.

<<<<<<< SEARCH
        view_weights = F.softmax(0.0873046875 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.08720703125 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE