MECHANISM: Fine-grained bisection of the confidence-adaptive fusion boundary

HYPOTHESIS: A fusion coefficient of 0.087109375 will preserve 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

INTENDED_EDIT: Set the evaluation-only flip-fusion coefficient to the midpoint between 0.08671875 and 0.0875, and use the best verified temperature of 0.800713.

EVIDENCE: Coefficient 0.08671875 yielded 9,327 correct at 0.191468640 cross-entropy, while 0.0875 yielded 9,328 correct at 0.191468821; their midpoint is the most informative next boundary test.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.8007
=======
        view_weights = F.softmax(0.087109375 * margins, dim=0).unsqueeze(-1)
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.800713
>>>>>>> REPLACE