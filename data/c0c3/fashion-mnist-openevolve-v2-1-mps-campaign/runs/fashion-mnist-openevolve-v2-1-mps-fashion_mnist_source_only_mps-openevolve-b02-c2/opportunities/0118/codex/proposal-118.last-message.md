MECHANISM: Validation-only top-four channel saliency

HYPOTHESIS: Evaluating the reliably trained hard-maximum model with top-four channel saliency will exceed 9,322 correct predictions by capturing the inference-side benefit of the best reference without its training-time ranking overhead.

INTENDED_EDIT: Retain global-maximum attention throughout training, but replace it with exact top-four averaging during evaluation.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322, while hard-maximum training reliably finished with 9,320; repeated training-time top-four variants timed out, motivating an evaluation-only isolation of the descriptor.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        if self.training:
            channel_salient = F.adaptive_max_pool2d(features, 1)
            channel_salient = channel_salient.flatten(2).transpose(1, 2)
        else:
            channel_salient = features.flatten(2).topk(
                4, dim=2, sorted=False
            ).values
            channel_salient = channel_salient.mean(dim=2, keepdim=True)
            channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE