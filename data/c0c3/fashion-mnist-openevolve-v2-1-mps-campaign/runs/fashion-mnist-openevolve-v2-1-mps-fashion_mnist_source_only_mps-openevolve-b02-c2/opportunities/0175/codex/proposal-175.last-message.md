MECHANISM: Bisection of confidence-adaptive flip-fusion boundary

HYPOTHESIS: A fusion-margin coefficient of 0.08125 will retain 9,328 correct predictions while lowering validation cross-entropy below 0.191468821.

INTENDED_EDIT: Reduce the evaluation-only fusion coefficient to the midpoint between 0.075 and 0.0875, and use the best verified temperature of 0.800713.

EVIDENCE: Coefficient 0.075 produced 9,327 correct, while 0.0875 produced 9,328 correct with lower cross-entropy than 0.10; their midpoint efficiently searches for the smallest coefficient preserving the additional correct prediction.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.8007
=======
        view_weights = F.softmax(0.08125 * margins, dim=0).unsqueeze(-1)
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.800713
>>>>>>> REPLACE