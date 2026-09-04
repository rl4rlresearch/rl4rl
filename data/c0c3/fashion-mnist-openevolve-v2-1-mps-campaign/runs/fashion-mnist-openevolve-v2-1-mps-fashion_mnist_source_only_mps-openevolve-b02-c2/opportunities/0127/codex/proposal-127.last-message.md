MECHANISM: Temperature-calibrated top-four channel saliency

HYPOTHESIS: Restoring the qualified top-four descriptor will recover 9,322 correct predictions, while temperature 1.05 will preserve those predictions and reduce cross-entropy below 0.192551.

INTENDED_EDIT: Use mean top-four channel evidence, restore ordinary BatchNorm momentum, and mildly soften the arithmetic-probability ensemble.

EVIDENCE: Reference Design 2 is the only verified implementation reaching 9,322 correct; the earlier calibrated top-four experiment timed out, so isolating calibration on the exact qualified implementation is the most informative retry.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        return (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        ) / 1.05
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