MECHANISM: Margin-weighted flip probability ensemble

HYPOTHESIS: Mildly weighting each validation orientation by its top-two log-probability margin will exceed 9,322 correct predictions by favoring the more decisive view on flip disagreements while preserving the qualified top-four model and training runtime.

INTENDED_EDIT: Replace equal arithmetic-probability flip fusion with a normalized, modestly confidence-weighted probability mixture during evaluation only.

EVIDENCE: Top-four saliency produced the best verified count of 9,322, and arithmetic probability fusion improved cross-entropy over mean-logit fusion without changing the hard-maximum model’s count; this isolates reliability-aware probability fusion without disturbing successful training.

<<<<<<< SEARCH
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        top_two = log_probabilities.topk(2, dim=2).values
        margins = top_two[:, :, 0] - top_two[:, :, 1]
        view_weights = F.softmax(0.25 * margins, dim=0).unsqueeze(-1)
        return torch.logsumexp(
            log_probabilities + torch.log(view_weights),
            dim=0,
        )
>>>>>>> REPLACE