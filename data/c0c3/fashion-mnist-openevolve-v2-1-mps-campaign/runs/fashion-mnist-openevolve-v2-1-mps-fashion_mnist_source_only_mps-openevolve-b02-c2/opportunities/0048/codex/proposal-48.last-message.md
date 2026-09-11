MECHANISM: Peak-tempered shared channel attention

HYPOTHESIS: Scaling the global-maximum descriptor by 0.80 while restoring plain paired-view cross-entropy will exceed 9,322 correct predictions by retaining fast peak detection while reducing the outlier dominance that top-four averaging appears to improve.

INTENDED_EDIT: Attenuate maximum-channel evidence before the shared attention kernel and remove the harmful ensemble-aware training loss.

EVIDENCE: Hard maximum achieved 9,320 correct with lower cross-entropy and faster training, while top-four averaging reached 9,322; local-patch averaging fell to 9,312. Peak attenuation tests whether top-four’s advantage comes from softened peak magnitude without incurring top-k’s timeout risk.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = 0.80 * channel_maximum
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
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