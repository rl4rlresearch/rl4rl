MECHANISM: Evaluation-only top-four channel saliency

HYPOTHESIS: Training with fast global-maximum attention but replacing it with top-four averaging during evaluation will exceed 9,322 correct predictions while retaining the completed hard-maximum model’s training runtime.

INTENDED_EDIT: Keep global-maximum channel evidence during training and compute the more robust top-four descriptor only during validation.

EVIDENCE: Top-four saliency achieved the best verified result at 9,322 correct versus 9,320 for hard maximum, but repeated top-k training variants exceeded the time limit; restricting top-k to evaluation preserves its inference benefit without adding training overhead.

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
            channel_salient = features.flatten(2).topk(4, dim=2).values
            channel_salient = channel_salient.mean(dim=2, keepdim=True)
            channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE