MECHANISM: Calibrated single-view cosine-tail refinement

HYPOTHESIS: Using single-view training during the final 10% of the near-zero-learning-rate cosine tail will prevent timeout while retaining at least 9,322 correct predictions, and temperature 1.05 will lower cross-entropy enough to exceed 9322.419269.

INTENDED_EDIT: Preserve exact top-four attention, halve forward work in the final cosine tail with batch-size-adjusted BatchNorm momentum, and mildly soften only the validation flip-mixture logits.

EVIDENCE: Top-four attention produced the best verified count of 9,322 but required 78.8 seconds and repeatedly timed out afterward; the reliable hard-maximum design finished near 75 seconds, while arithmetic probability ensembling reduced cross-entropy at unchanged accuracy. The final 10% contributes less than 0.4% of the cosine schedule’s integrated learning rate.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        mixture_logits = torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        return mixture_logits / 1.05
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    progress = step / max(total_steps, 1)
    if progress >= 0.90:
        half_batch_momentum = 1.0 - (1.0 - 0.1) ** 0.5
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = half_batch_momentum
        return F.cross_entropy(model(images), labels)

    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
>>>>>>> REPLACE