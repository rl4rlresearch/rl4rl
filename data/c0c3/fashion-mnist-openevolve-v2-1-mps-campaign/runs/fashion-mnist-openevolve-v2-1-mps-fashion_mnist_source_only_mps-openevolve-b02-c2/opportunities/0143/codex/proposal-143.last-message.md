MECHANISM: Straight-through top-four channel saliency with temperature calibration

HYPOTHESIS: Preserving top-four saliency in the forward pass while routing its gradient through the inexpensive hard maximum will finish within the time limit and achieve at least 9,322 correct predictions; temperature 1.05 will then reduce cross-entropy below 0.192551 without changing predictions.

INTENDED_EDIT: Compute unsorted top-four evidence from detached features, use hard-max evidence as its straight-through gradient, fuse the bias-free attention calls, and soften only the final ensemble logits.

EVIDENCE: Exact top-four saliency produced the best verified count of 9,322 but repeatedly timed out, whereas hard-max attention reliably finished with 9,320; this retains the successful top-four forward statistic while eliminating its ranking backward pass.

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
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_salient = features.detach().flatten(2).topk(
            4, dim=2, sorted=False
        ).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_salient = channel_maximum + (
            channel_salient - channel_maximum
        ).detach()
        channel_gate = self.channel_attention(
            channel_average + channel_salient
        )
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        ensemble_logits = (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        )
        return ensemble_logits / 1.05
>>>>>>> REPLACE