MECHANISM: Two-mask dropout consistency regularization

HYPOTHESIS: Averaging supervised loss across two classifier dropout masks and softly aligning their predictions will exceed 9,330 correct predictions by reducing dropout-induced classifier variance without perturbing the position-sensitive image representation.

INTENDED_EDIT: Run the inexpensive dense classifier twice during training and add a modest symmetric-KL consistency penalty while retaining the proven dropout, smoothing, optimizer, EMA, and inference behavior.

EVIDENCE: Spatial changes underperformed—translation augmentation reached 9,208, classifier widening 9,301, and local refinement 9,309—while the evidence identifies dropout’s linear taper as beneficial; this tests stronger use of that regularizer without changing spatial evidence or parameter count.

<<<<<<< SEARCH
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        return self.classifier(fused)
=======
        fused = self.view_fusion(
            torch.cat((invariant, disagreement), dim=1)
        )
        logits = self.classifier(fused)
        if self.training:
            self.dropout_consistency_logits = self.classifier(fused)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
=======
    logits = model(images)
    second_logits = model.dropout_consistency_logits
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    supervised_loss = 0.5 * (
        F.cross_entropy(
            logits,
            labels,
            label_smoothing=label_smoothing,
        )
        + F.cross_entropy(
            second_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    )
    log_probs = F.log_softmax(logits, dim=1)
    second_log_probs = F.log_softmax(second_logits, dim=1)
    symmetric_kl = 0.5 * (
        F.kl_div(
            log_probs,
            second_log_probs,
            reduction="batchmean",
            log_target=True,
        )
        + F.kl_div(
            second_log_probs,
            log_probs,
            reduction="batchmean",
            log_target=True,
        )
    )
    return supervised_loss + 0.25 * symmetric_kl
>>>>>>> REPLACE