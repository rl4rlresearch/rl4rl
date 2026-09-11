MECHANISM: Late-phase exact top-four saliency curriculum

HYPOTHESIS: Switching from hard-maximum to exact top-four channel saliency halfway through training will exceed 9,322 correct predictions while avoiding the runtime failures of full-training top-four selection.

INTENDED_EDIT: Train the first half with fast maximum saliency, fine-tune the second half and evaluate with unsorted exact top-four saliency, eliminate the redundant preparatory flip, fuse the linear attention calls, and use arithmetic probability averaging for validation.

EVIDENCE: Reference Design 3 achieved the best verified count of 9,322 using top-four saliency, while hard-maximum designs reliably completed around 75–78 seconds with 9,320; repeated full-training top-four variants timed out, motivating a half-duration curriculum that retains late adaptation while reducing selection overhead.

<<<<<<< SEARCH
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)

        self.classifier = nn.Sequential(
=======
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
        self.use_top_four_saliency = False

        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_average = features.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        if self.training and not self.use_top_four_saliency:
            channel_salient = features.amax(
                dim=(2, 3), keepdim=False
            ).unsqueeze(1)
        else:
            channel_salient = torch.topk(
                features.flatten(2),
                k=4,
                dim=2,
                sorted=False,
            ).values.mean(dim=2, keepdim=False).unsqueeze(1)
        channel_gate = self.channel_attention(
            channel_average + channel_salient
        )
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
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

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
=======
) -> torch.Tensor:
    model.use_top_four_saliency = 2 * step >= total_steps
    paired_images = torch.cat(
>>>>>>> REPLACE