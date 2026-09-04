MECHANISM: Loss-aligned geometric flip ensembling

HYPOTHESIS: Plain paired-view cross-entropy with mean-logit flip ensembling will exceed 9,322 correct predictions while retaining the reliably completed hard-maximum architecture’s runtime.

INTENDED_EDIT: Remove the harmful ensemble-aware training loss and replace arithmetic probability averaging at validation with geometric probability averaging implemented as mean logits.

EVIDENCE: Plain paired-view cross-entropy reached 9,320 correct versus 9,307 for the ensemble-aware objective; the prior geometric-ensemble test used the timeout-prone top-four model, so testing it on the 75.3-second hard-maximum design isolates the inference rule with runtime headroom.

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
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(paired_logits, paired_labels)

    batch_size = images.shape[0]
    ensemble_log_probabilities = torch.logsumexp(
        torch.stack(
            (
                F.log_softmax(paired_logits[:batch_size], dim=1),
                F.log_softmax(paired_logits[batch_size:], dim=1),
            )
        ),
        dim=0,
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.5 * (individual_loss + ensemble_loss)
=======
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
>>>>>>> REPLACE