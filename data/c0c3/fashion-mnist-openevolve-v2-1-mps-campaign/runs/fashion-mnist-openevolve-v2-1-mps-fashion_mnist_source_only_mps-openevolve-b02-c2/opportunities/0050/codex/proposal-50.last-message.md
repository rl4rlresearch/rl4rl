MECHANISM: Unsorted top-four channel saliency

HYPOTHESIS: Averaging each channel’s four strongest activations without sorting them will exceed the current 9,320 correct predictions, match or improve the verified 9,322-correct top-four design, and reduce its runtime overhead.

INTENDED_EDIT: Replace hard global-maximum attention with top-four averaging while disabling the unnecessary ordering of selected activations.

EVIDENCE: Reference Design 2 achieved the best result, 9,322 correct, using top-four saliency; hard maximum reached 9,320. Recent top-k variants approached the time limit, while sorting is irrelevant before averaging.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_salient = features.flatten(2).topk(
            4, dim=2, sorted=False
        ).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE