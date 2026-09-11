MECHANISM: Axis-stratified upper-tail channel saliency

HYPOTHESIS: Blending the global maximum with mean row and column maxima will exceed 9,322 correct predictions by approximating distributed top-four evidence without ranking overhead or the locality bias of 2×2 pooling.

INTENDED_EDIT: Replace hard-maximum channel evidence with a 50/25/25 global/row/column maximum blend and restore ordinary BatchNorm momentum.

EVIDENCE: Exact top-four saliency achieved 9,322 correct versus 9,320 for hard maxima, while contiguous 2×2 pooling fell to 9,297; ordinary BatchNorm also ran faster with slightly lower cross-entropy than sample-weighted updates.

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        row_maxima = features.amax(dim=3)
        column_maxima = features.amax(dim=2)
        channel_salient = (
            0.50 * row_maxima.amax(dim=2)
            + 0.25 * row_maxima.mean(dim=2)
            + 0.25 * column_maxima.mean(dim=2)
        ).unsqueeze(1)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
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