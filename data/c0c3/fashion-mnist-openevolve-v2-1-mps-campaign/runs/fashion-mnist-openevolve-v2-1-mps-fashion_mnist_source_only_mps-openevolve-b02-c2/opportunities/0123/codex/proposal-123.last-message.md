MECHANISM: Late top-four saliency curriculum

HYPOTHESIS: Training with inexpensive hard-maximum attention for the first 40% of steps, then adapting under exact top-four saliency for the remaining 60%, will exceed 9,322 correct predictions while avoiding the timeout risk of full-run top-four pooling.

INTENDED_EDIT: Restore ordinary BatchNorm momentum and switch channel attention from global maxima to exact top-four evidence after 40% of training; evaluation always uses top-four saliency.

EVIDENCE: Full-run top-four saliency achieved the best verified count of 9,322 but took 78.8 seconds, while hard-maximum training finished faster at 9,320; evaluation-only switching regressed to 9,315, showing that the model needs training exposure to the top-four descriptor.

<<<<<<< SEARCH
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)

        self.classifier = nn.Sequential(
=======
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
        self.use_top_four_saliency = True

        self.classifier = nn.Sequential(
>>>>>>> REPLACE

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
        if self.use_top_four_saliency:
            channel_salient = features.flatten(2).topk(4, dim=2).values
            channel_salient = channel_salient.mean(dim=2, keepdim=True)
            channel_salient = channel_salient.transpose(1, 2)
        else:
            channel_salient = F.adaptive_max_pool2d(features, 1)
            channel_salient = channel_salient.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum

    paired_images = torch.cat(
=======
) -> torch.Tensor:
    model.use_top_four_saliency = step >= 0.4 * total_steps
    paired_images = torch.cat(
>>>>>>> REPLACE