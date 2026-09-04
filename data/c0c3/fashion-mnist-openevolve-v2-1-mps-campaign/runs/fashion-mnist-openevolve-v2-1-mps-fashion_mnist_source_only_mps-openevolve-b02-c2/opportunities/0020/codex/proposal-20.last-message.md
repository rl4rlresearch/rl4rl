MECHANISM: Peak-preserving top-four channel attention

HYPOTHESIS: Blending the spatial maximum with the top-four mean will exceed 9,322 correct predictions by retaining strong peak evidence while reducing single-activation sensitivity.

INTENDED_EDIT: Remove the harmful translation augmentation and replace hard-maximum channel evidence with an equal blend of the maximum and top-four mean, using the existing shared attention kernel.

EVIDENCE: Top-four channel attention achieved 9,322 correct versus 9,320 for hard maxima, while hard maxima had slightly lower cross-entropy; the blend isolates a middle ground between their saliency profiles. Translation augmentation fell sharply to 9,178 correct.

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
        channel_values = features.flatten(2).topk(4, dim=2).values
        channel_salient = 0.5 * (
            channel_values[:, :, :1]
            + channel_values.mean(dim=2, keepdim=True)
        )
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    height, width = images.shape[-2:]
    images = F.pad(images, (2, 2, 2, 2), mode="replicate")
    offsets = torch.randint(
        0,
        5,
        (images.shape[0], 2),
        device=images.device,
    )

    row_indices = offsets[:, :1] + torch.arange(height, device=images.device)
    images = images.gather(
        2,
        row_indices[:, None, :, None].expand(
            -1, images.shape[1], -1, images.shape[3]
        ),
    )
    column_indices = offsets[:, 1:] + torch.arange(width, device=images.device)
    images = images.gather(
        3,
        column_indices[:, None, None, :].expand(
            -1, images.shape[1], images.shape[2], -1
        ),
    )

    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
=======
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
>>>>>>> REPLACE