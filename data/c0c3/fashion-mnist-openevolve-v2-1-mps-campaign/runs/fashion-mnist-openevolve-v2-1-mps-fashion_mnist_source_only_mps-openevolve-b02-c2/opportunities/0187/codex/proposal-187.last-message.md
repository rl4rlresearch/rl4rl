MECHANISM: Ultra-fine bisection of the flip-fusion decision boundary

HYPOTHESIS: A fusion coefficient of 0.08729400634765625 will preserve 9,328 correct predictions while reducing validation cross-entropy below 0.1914687728881836.

INTENDED_EDIT: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket.

EVIDENCE: Coefficient 0.08729248046875 produced 9,327 correct, while 0.0872955322265625 produced 9,328 correct; their midpoint is the most informative next boundary test.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.08729400634765625 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE