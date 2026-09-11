MECHANISM: Mild per-view target smoothing with confidence-adaptive flip fusion

HYPOTHESIS: Applying 0.02 label smoothing to the reliable paired-view objective will exceed 9,322 correct predictions by reducing brittle class-boundary fitting, while the verified 0.10 confidence-adaptive fusion improves the cross-entropy tie-breaker.

INTENDED_EDIT: Add mild label smoothing without forcing flip agreement, and replace mean-logit evaluation with the best verified confidence-adaptive probability ensemble.

EVIDENCE: Hard-max attention reliably reached 9,320 while attention approximations, translation augmentation, and paired-mixture supervision regressed or timed out; coefficient-0.10 probability fusion retained all 9,320 correct and improved cross-entropy from 0.192650 to 0.192214.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        top_two = log_probabilities.topk(2, dim=2).values
        margins = top_two[..., 0] - top_two[..., 1]
        view_weights = F.softmax(0.10 * margins, dim=0).unsqueeze(-1)
        return torch.logsumexp(
            log_probabilities + view_weights.log(),
            dim=0,
        )
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