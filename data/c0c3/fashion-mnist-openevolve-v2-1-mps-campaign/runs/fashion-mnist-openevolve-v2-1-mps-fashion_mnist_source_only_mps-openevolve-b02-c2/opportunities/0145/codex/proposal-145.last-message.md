MECHANISM: Late-phase saliency curriculum

HYPOTHESIS: Gradually replacing hard-max channel evidence with top-four evidence during the final half of training will exceed 9,322 correct predictions while avoiding the repeated full-training top-four timeout.

INTENDED_EDIT: Train with inexpensive hard-max saliency for the first half, smoothly transition to top-four saliency during the second half, and use pure top-four saliency for validation.

EVIDENCE: Full top-four training achieved the best verified count of 9,322 but repeatedly timed out, whereas hard-max training reliably finished in 66.6–75.3 seconds with 9,320 correct; limiting top-four computation to late optimization preserves adaptation time while targeting a safer runtime.

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = features.flatten(2).topk(4, dim=2).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        saliency_mix = (
            1.0
            if not self.training
            else getattr(self, "saliency_mix", 0.0)
        )
        channel_maximum = features.amax(dim=(2, 3), keepdim=True)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        if saliency_mix > 0.0:
            channel_top_four = features.flatten(2).topk(4, dim=2).values
            channel_top_four = channel_top_four.mean(dim=2, keepdim=True)
            channel_top_four = channel_top_four.transpose(1, 2)
            channel_salient = torch.lerp(
                channel_maximum,
                channel_top_four,
                saliency_mix,
            )
        else:
            channel_salient = channel_maximum
        channel_gate = self.channel_attention(channel_average)
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
=======
) -> torch.Tensor:
    progress = step / max(total_steps, 1)
    model.saliency_mix = min(max(2.0 * progress - 1.0, 0.0), 1.0)
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
>>>>>>> REPLACE