MECHANISM: Flip-mixture-aware paired supervision

HYPOTHESIS: Adding a 25% validation-aligned arithmetic-probability mixture loss will exceed 9,322 correct predictions while retaining reliable hard-maximum attention runtime; probability averaging will also reduce cross-entropy if correct counts tie.

INTENDED_EDIT: Optimize a blend of per-view cross-entropy and paired flip-mixture NLL, and use the same arithmetic probability mixture during validation.

EVIDENCE: Arithmetic probability ensembling reduced hard-maximum cross-entropy from 0.1926495 to 0.1922617 without changing its 9,320 correct predictions, while numerous additional saliency and augmentation computations timed out or regressed; directly supervising that proven aggregation is an inexpensive orthogonal test.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(paired_logits, paired_labels)

    first_logits, second_logits = paired_logits.chunk(2, dim=0)
    mixture_log_probabilities = torch.logaddexp(
        F.log_softmax(first_logits, dim=1),
        F.log_softmax(second_logits, dim=1),
    ) - math.log(2.0)
    mixture_loss = F.nll_loss(mixture_log_probabilities, labels)
    return 0.75 * individual_loss + 0.25 * mixture_loss
>>>>>>> REPLACE