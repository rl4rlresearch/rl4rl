MECHANISM: Temperature-controlled smooth channel saliency

HYPOTHESIS: Replacing hard channel maxima with a temperature-0.25 log-mean-exp descriptor will exceed 9,322 correct predictions by approximating robust top-four saliency with dense gradients and without top-k runtime overhead.

INTENDED_EDIT: Remove the harmful translation augmentation and replace global-maximum channel evidence with a parameter-free smooth-maximum descriptor while preserving shared attention, paired-view training, and flip-ensemble validation.

EVIDENCE: Top-four channel attention achieved the best verified result at 9,322 correct versus 9,320 for hard maxima, but subsequent top-k variants timed out; translations reduced correctness to 9,178. A smooth reduction tests the same robust-saliency principle without sorting.

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
        spatial_features = features.flatten(2)
        temperature = 0.25
        channel_salient = temperature * (
            torch.logsumexp(spatial_features / temperature, dim=2, keepdim=True)
            - math.log(spatial_features.shape[2])
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