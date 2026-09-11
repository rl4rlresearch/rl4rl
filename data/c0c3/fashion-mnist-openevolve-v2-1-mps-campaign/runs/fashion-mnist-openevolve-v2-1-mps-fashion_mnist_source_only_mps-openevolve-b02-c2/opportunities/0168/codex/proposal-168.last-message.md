MECHANISM: Stronger confidence-adaptive flip fusion

HYPOTHESIS: Increasing the fusion-margin coefficient from 0.10 to 0.20 will preserve the reliable 9,328-correct constant-smoothing solution and correct additional flip-disagreement cases by giving the more decisive view greater influence.

INTENDED_EDIT: Restore constant 0.02 label smoothing and temperature 0.800713, while strengthening confidence-adaptive validation fusion to coefficient 0.20.

EVIDENCE: Constant 0.02 smoothing produced 9,328 correct, and coefficient-0.10 confidence fusion improved cross-entropy without reducing correct predictions; testing a stronger coefficient directly determines whether view confidence can also improve the primary correct-count objective.

<<<<<<< SEARCH
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
=======
        view_weights = F.softmax(0.20 * margins, dim=0).unsqueeze(-1)
        fused_log_probabilities = torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
        return fused_log_probabilities / 0.800713
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE