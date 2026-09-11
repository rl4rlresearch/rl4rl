MECHANISM: Confidence-adaptive flip probability fusion

HYPOTHESIS: Mildly favoring the more decisive horizontal view will finish within the reliable hard-maximum runtime and exceed 9,322 correct predictions by resolving flip disagreements more accurately.

INTENDED_EDIT: Restore ordinary BatchNorm momentum and replace equal validation-view averaging with a normalized, modestly margin-weighted probability mixture.

EVIDENCE: Reference Design 1 reached 9,320 correct faster and with slightly lower cross-entropy than sample-weighted BatchNorm; confidence-weighted fusion remains untested because its previous attempt was coupled to repeatedly timing-out top-four attention.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
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
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum

    paired_images = torch.cat(
=======
    del step, total_steps
    paired_images = torch.cat(
>>>>>>> REPLACE