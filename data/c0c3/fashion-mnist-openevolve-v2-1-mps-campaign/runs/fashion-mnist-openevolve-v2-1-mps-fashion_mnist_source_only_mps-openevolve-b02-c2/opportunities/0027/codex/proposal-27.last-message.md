MECHANISM: Probability-ensemble-aware flip-pair training

HYPOTHESIS: Training the proven 9,320-correct shared average-plus-maximum attention model with a balanced individual-view and probability-ensemble loss will exceed 9,322 correct predictions by directly optimizing the same arithmetic flip ensemble used during validation while retaining supervision for both orientations.

INTENDED_EDIT: Restore shared global-average-plus-maximum channel attention and replace plain paired-view cross-entropy with an equal blend of individual-view cross-entropy and validation-matched flip-ensemble negative log-likelihood.

EVIDENCE: Shared average-plus-maximum attention reached 9,320 correct and paired-view probability ensembling was beneficial; top-four attention improved by only two additional correct predictions but repeatedly encountered runtime failures, motivating a computationally light objective-level refinement on the reliable 9,320-correct design.

<<<<<<< SEARCH
        channel_summary = F.adaptive_avg_pool2d(features, 1)
        channel_summary = channel_summary.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_summary)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
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
>>>>>>> REPLACE