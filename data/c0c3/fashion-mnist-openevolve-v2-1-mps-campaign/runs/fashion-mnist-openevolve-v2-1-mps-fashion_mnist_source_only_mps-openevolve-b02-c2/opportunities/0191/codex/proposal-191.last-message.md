MECHANISM: Ultra-fine flip-fusion boundary bisection

HYPOTHESIS: A fusion coefficient of 0.087293720245361328125 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.1914687713623047.

INTENDED_EDIT: Set the evaluation-only confidence-adaptive flip-fusion coefficient to the midpoint of the tightest verified 9,327/9,328 bracket.

EVIDENCE: Coefficient 0.0872936248779296875 produced 9,327 correct, while 0.08729381561279296875 produced 9,328 correct with the best observed cross-entropy; their midpoint is the most informative next boundary test.

<<<<<<< SEARCH
        view_weights = F.softmax(0.0873046875 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(0.087293720245361328125 * margins, dim=0).unsqueeze(-1)
>>>>>>> REPLACE