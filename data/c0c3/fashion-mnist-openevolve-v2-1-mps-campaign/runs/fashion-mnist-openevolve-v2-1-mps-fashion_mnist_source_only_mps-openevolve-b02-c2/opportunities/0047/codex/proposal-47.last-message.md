MECHANISM: Geometric flip ensemble on fast maximum attention

HYPOTHESIS: Averaging paired-orientation logits with the 9,320-correct hard-maximum model will exceed 9,322 correct predictions by preventing one orientation’s overconfident probability from dominating the ensemble.

INTENDED_EDIT: Restore global-maximum channel saliency and replace arithmetic probability averaging at validation with equal logit averaging.

EVIDENCE: Hard-maximum attention reached 9,320 correct with lower cross-entropy and faster training than the 9,322-correct top-four model; ensemble-aware training hurt accuracy, motivating an inference-only aggregation change that preserves ordinary paired-view cross-entropy.

<<<<<<< SEARCH
        channel_salient = F.avg_pool2d(features, kernel_size=2, stride=1)
        channel_salient = F.adaptive_max_pool2d(channel_salient, 1)
        channel_salient = channel_salient.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
=======
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
>>>>>>> REPLACE

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE