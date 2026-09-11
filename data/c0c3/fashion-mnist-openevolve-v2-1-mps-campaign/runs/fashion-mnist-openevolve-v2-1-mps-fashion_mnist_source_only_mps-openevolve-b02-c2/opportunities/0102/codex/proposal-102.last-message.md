MECHANISM: Validation-aligned paired-view mixture supervision

HYPOTHESIS: Blending ordinary per-view cross-entropy with arithmetic flip-mixture negative log-likelihood will exceed 9,322 correct predictions by directly optimizing the ensemble used during validation without additional model passes.

INTENDED_EDIT: Reuse the paired logits to combine equal-weight individual-view and probability-mixture losses.

EVIDENCE: Arithmetic probability flip ensembling reduced cross-entropy from 0.1926495 to 0.1922617 at the same 9,320 correct predictions; aligning training with that beneficial validation rule is an orthogonal, low-overhead change to the reliable hard-maximum design.

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    paired_logits = model(paired_images).reshape(2, images.shape[0], -1)
    paired_log_probabilities = F.log_softmax(paired_logits, dim=2)
    individual_loss = F.nll_loss(
        paired_log_probabilities.flatten(0, 1),
        labels.repeat(2),
    )
    mixture_log_probabilities = (
        torch.logsumexp(paired_log_probabilities, dim=0) - math.log(2.0)
    )
    mixture_loss = F.nll_loss(mixture_log_probabilities, labels)
    return 0.5 * (individual_loss + mixture_loss)
>>>>>>> REPLACE