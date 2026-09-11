MECHANISM: Redundancy-free top-four channel saliency

HYPOTHESIS: Restoring the 9,322-correct top-four attention while removing the functionally redundant preparation flip will achieve at least 9,323 correct predictions and finish within the time limit.

INTENDED_EDIT: Replace global-maximum channel evidence with top-four averaging, and pass training batches through unchanged because the loss already evaluates both horizontal orientations.

EVIDENCE: Top-four saliency produced the best verified result at 9,322 correct; the recent translation trial established that preparation-time flipping is redundant with paired-view training, so removing it provides runtime headroom for the stronger attention reduction.

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
    del step, total_steps
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        torch.flip(images, dims=(-1,)),
        images,
    )
    return images, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE