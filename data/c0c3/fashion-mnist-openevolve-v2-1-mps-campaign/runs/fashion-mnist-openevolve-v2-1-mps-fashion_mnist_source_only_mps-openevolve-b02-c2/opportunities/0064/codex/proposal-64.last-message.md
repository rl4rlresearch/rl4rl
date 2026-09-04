MECHANISM: Reduced classifier dropout with plain paired-view supervision

HYPOTHESIS: Reducing classifier dropout from 0.10 to 0.05 while restoring plain paired-view cross-entropy will exceed 9,322 correct predictions by improving fixed-exposure fitting without removing stochastic regularization.

INTENDED_EDIT: Remove the harmful ensemble-aware loss and halve classifier dropout, preserving the fast hard-maximum attention architecture and all optimizer settings.

EVIDENCE: Plain paired-view cross-entropy with hard-maximum attention reached 9,320 correct and 0.19226 cross-entropy, while the ensemble-aware objective fell to 9,307; the remaining two-prediction gap motivates a low-cost reduction in training-time regularization.

<<<<<<< SEARCH
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
=======
            nn.GELU(),
            nn.Dropout(p=0.05),
            nn.Linear(48, 10),
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