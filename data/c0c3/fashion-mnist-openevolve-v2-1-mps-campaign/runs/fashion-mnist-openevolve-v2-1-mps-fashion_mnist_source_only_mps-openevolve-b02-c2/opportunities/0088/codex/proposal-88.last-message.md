MECHANISM: Progressive hard-max-to-top-four channel saliency

HYPOTHESIS: Smoothly introducing top-four saliency during the final half of training will exceed 9,322 correct predictions while avoiding the repeated runtime failures of using top-k throughout training.

INTENDED_EDIT: Train initially with fast hard-maximum attention, linearly transition to exact top-four attention over the final half, use top-four attention at evaluation, and restore arithmetic probability flip ensembling.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322, while hard maxima reliably completed at 9,320 and later full-training top-k variants repeatedly timed out; arithmetic probability ensembling also had lower cross-entropy than mean-logit ensembling at the same hard-max correct count.

<<<<<<< SEARCH
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)

        self.classifier = nn.Sequential(
=======
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
        self.saliency_blend = 0.0

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
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        blend = self.saliency_blend if self.training else 1.0
        if blend > 0.0:
            channel_salient = features.flatten(2).topk(4, dim=2).values
            channel_salient = channel_salient.mean(dim=2, keepdim=True)
            channel_salient = channel_salient.transpose(1, 2)
            channel_salient = torch.lerp(channel_maximum, channel_salient, blend)
        else:
            channel_salient = channel_maximum
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
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
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
=======
) -> torch.Tensor:
    model.saliency_blend = max(
        0.0,
        min(1.0, 2.0 * step / max(total_steps, 1) - 1.0),
    )
    paired_images = torch.cat(
>>>>>>> REPLACE