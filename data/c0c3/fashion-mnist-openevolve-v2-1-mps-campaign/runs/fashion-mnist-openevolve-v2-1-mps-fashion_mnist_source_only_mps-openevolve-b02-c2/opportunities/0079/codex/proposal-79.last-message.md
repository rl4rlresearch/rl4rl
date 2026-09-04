MECHANISM: Local four-pixel peak pooling

HYPOTHESIS: A maximum over overlapping 2×2 feature averages will exceed 9,322 correct predictions by preserving the successful four-activation saliency effect while favoring spatially coherent evidence and avoiding top-k’s runtime cost.

INTENDED_EDIT: Replace hard channel maxima with peak 2×2 average pooling, fuse the bias-free attention calls, and restore ordinary BatchNorm momentum.

EVIDENCE: Top-four saliency achieved the best result at 9,322 correct, but later top-k implementations timed out; ordinary BatchNorm matched sample-weighted BatchNorm at 9,320 correct while finishing faster.

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = F.avg_pool2d(features, kernel_size=2, stride=1)
        channel_salient = F.adaptive_max_pool2d(channel_salient, 1)
        channel_salient = channel_salient.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average + channel_salient)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
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