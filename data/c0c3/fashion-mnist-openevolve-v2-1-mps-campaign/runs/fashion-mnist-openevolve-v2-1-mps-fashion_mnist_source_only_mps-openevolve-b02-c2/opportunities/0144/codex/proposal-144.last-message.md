MECHANISM: Row-compressed top-four channel saliency

HYPOTHESIS: Selecting the four strongest row maxima per channel will finish within the reliable runtime envelope and exceed 9,322 correct predictions by retaining robust upper-tail evidence while reducing each top-k operation from 49 candidates to 7.

INTENDED_EDIT: Replace global hard-max attention with top-four saliency over row maxima, and use the verified equal probability-space flip ensemble.

EVIDENCE: Exact spatial top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard maxima reliably finished at 9,320; compressing each channel to seven row maxima preserves salient, horizontally flip-invariant candidates with substantially less ranking work.

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
        channel_salient = features.amax(dim=3)
        channel_salient = channel_salient.topk(
            4, dim=2, sorted=False
        ).values
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