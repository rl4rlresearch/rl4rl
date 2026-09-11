MECHANISM: RMS-tempered channel saliency

HYPOTHESIS: Blending global maxima with per-channel RMS activations and restoring plain paired-view cross-entropy will exceed 9,322 correct predictions while avoiding top-k’s runtime cost.

INTENDED_EDIT: Replace hard-maximum channel evidence with a fast equal blend of maximum and RMS descriptors, and remove the ensemble-aware loss that reduced accuracy.

EVIDENCE: Top-four saliency improved hard maximum from 9,320 to 9,322 correct, suggesting reduced outlier sensitivity, while repeated top-k variants timed out; plain paired-view cross-entropy reached 9,320 whereas the current ensemble-aware objective fell to 9,307.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_rms = F.adaptive_avg_pool2d(features.square(), 1).sqrt()
        channel_salient = 0.5 * (channel_maximum + channel_rms)
        channel_salient = channel_salient.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
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