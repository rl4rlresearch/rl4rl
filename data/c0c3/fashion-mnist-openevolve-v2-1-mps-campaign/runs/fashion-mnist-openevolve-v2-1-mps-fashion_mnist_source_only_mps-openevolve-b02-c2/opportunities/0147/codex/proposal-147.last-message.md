MECHANISM: Spatially balanced four-region channel saliency

HYPOTHESIS: Averaging four regional maxima will finish within the reliable runtime envelope and exceed 9,322 correct predictions by approximating top-four saliency without a costly ranking operation.

INTENDED_EDIT: Replace each channel’s single global maximum with the mean of a 2×2 adaptive-max grid, leaving training and evaluation otherwise unchanged.

EVIDENCE: Global top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard maxima reliably reached 9,320; regional max pooling supplies four robust salient activations using the already-reliable pooling primitive instead of `topk`.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_salient = F.adaptive_max_pool2d(features, 2)
        channel_salient = channel_salient.flatten(2).mean(
            dim=2, keepdim=True
        )
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE