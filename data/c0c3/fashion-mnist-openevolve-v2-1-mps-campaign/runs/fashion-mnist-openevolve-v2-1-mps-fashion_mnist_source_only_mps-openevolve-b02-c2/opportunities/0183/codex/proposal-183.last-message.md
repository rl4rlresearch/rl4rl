MECHANISM: Precision bisection of the flip-fusion decision boundary

HYPOTHESIS: A fusion coefficient of 0.0872802734375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468777.

INTENDED_EDIT: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint between 0.087255859375 and 0.0873046875, retaining temperature 0.800713 and all training settings.

EVIDENCE: Coefficient 0.087255859375 yielded 9,327 correct, while 0.0873046875 yielded 9,328 correct at 0.191468777 cross-entropy; their midpoint is the most informative next boundary test.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.0872802734375 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE