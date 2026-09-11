MECHANISM: Iterative-reduction exact top-four channel saliency

HYPOTHESIS: Exact top-four saliency computed without `topk` will finish reliably and recover at least 9,322 correct predictions, exceeding the current 9,320-correct design.

INTENDED_EDIT: Restore top-four attention using four max-and-scatter reductions, fuse the bias-free attention calls, restore arithmetic probability flip ensembling, and remove the redundant preparatory flip.

EVIDENCE: Reference Design 3 reached 9,322 correct with top-four saliency, while later `topk` variants timed out; the reduction-based top-two implementation finished in 71.8 seconds, showing this computational route avoids the problematic sorting overhead.

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

        remaining = features.flatten(2)
        salient_sum = 0.0
        for _ in range(4):
            salient_value, salient_index = remaining.max(dim=2, keepdim=True)
            salient_sum = salient_sum + salient_value
            remaining = remaining.scatter(
                2, salient_index, float("-inf")
            )
        channel_salient = (0.25 * salient_sum).transpose(1, 2)

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