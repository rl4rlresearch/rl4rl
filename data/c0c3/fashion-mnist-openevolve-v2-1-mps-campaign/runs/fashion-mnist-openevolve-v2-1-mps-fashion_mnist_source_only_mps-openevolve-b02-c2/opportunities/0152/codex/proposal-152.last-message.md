MECHANISM: Evaluation-only top-four channel saliency

HYPOTHESIS: Using top-four saliency only during validation will reach at least 9,322 correct predictions without the full-training timeout, while probability-space flip fusion will improve the cross-entropy tie-breaker.

INTENDED_EDIT: Preserve the reliable hard-maximum training path, switch to exact top-four channel evidence only in evaluation mode, and average flip predictions in probability space.

EVIDENCE: Exact top-four saliency produced the best verified count of 9,322 but repeatedly timed out when used during training; hard-max training reliably finished at 9,320, and probability fusion reduced its cross-entropy from 0.192650 to 0.192262 without reducing correct predictions.

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
        if self.training:
            channel_salient = F.adaptive_max_pool2d(features, 1)
            channel_salient = channel_salient.flatten(2).transpose(1, 2)
        else:
            channel_salient = features.flatten(2).topk(4, dim=2).values
            channel_salient = channel_salient.mean(dim=2, keepdim=True)
            channel_salient = channel_salient.transpose(1, 2)
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