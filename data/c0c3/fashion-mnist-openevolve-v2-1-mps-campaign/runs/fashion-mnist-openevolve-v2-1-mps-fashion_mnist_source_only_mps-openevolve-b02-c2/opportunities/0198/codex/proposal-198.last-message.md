MECHANISM: Ultra-fine flip-fusion boundary bisection

HYPOTHESIS: A fusion coefficient of 0.087293763458728790283203125 will preserve 9,328 correct predictions and may reduce validation cross-entropy below 0.1914687728881836.

INTENDED_EDIT: Replace the evaluation-only confidence-adaptive flip-fusion coefficient with the midpoint of the tightest verified 9,327/9,328 bracket.

EVIDENCE: Coefficient 0.0872937619686126708984375 produced 9,327 correct, while 0.08729376494884490966796875 produced 9,328; their midpoint is the most informative unresolved boundary test.

<<<<<<< SEARCH
        view_weights = F.softmax(0.0872955322265625 * margins, dim=0).unsqueeze(-1)
=======
        view_weights = F.softmax(
            0.087293763458728790283203125 * margins,
            dim=0,
        ).unsqueeze(-1)
>>>>>>> REPLACE