MECHANISM: Probability-space mirrored-view ensemble

HYPOTHESIS: Arithmetic averaging of view probabilities will exceed 9,286 correct predictions by allowing either orientation to carry uncertain examples while preserving direct optimization of the mirrored ensemble.

INTENDED_EDIT: Replace geometric probability averaging induced by mean logits with a normalized arithmetic probability mixture during both training and validation.

EVIDENCE: Paired-view training improved correct predictions from 9,237 to 9,282, while auxiliary per-view supervision regressed to 9,252; this suggests the gain comes from complementary ensemble behavior rather than forcing each view to succeed independently.

<<<<<<< SEARCH
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return torch.logaddexp(
            F.log_softmax(logits, dim=1),
            F.log_softmax(flipped_logits, dim=1),
        ) - math.log(2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_logits = model(paired_images)
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
    paired_logits = model(paired_images)
    logits = torch.logaddexp(
        F.log_softmax(paired_logits[:batch_size], dim=1),
        F.log_softmax(paired_logits[batch_size:], dim=1),
    ) - math.log(2.0)
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
>>>>>>> REPLACE