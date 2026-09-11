MECHANISM: Stop-gradient top-four channel saliency

HYPOTHESIS: Exact top-four saliency with its sparse backbone gradient detached will exceed 9,322 correct predictions while avoiding the runtime failures of later top-k variants.

INTENDED_EDIT: Restore unsorted top-four channel evidence, stop its gradient into the feature backbone, and algebraically fuse the two bias-free attention calls.

EVIDENCE: Reference Design 2 achieved the best verified count of 9,322 with exact top-four saliency; repeated top-k variants timed out, motivating preservation of its forward evidence while removing selection backward work and one redundant attention invocation.

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
        channel_salient = features.detach().flatten(2).topk(
            4, dim=2, sorted=False
        ).values
        channel_salient = channel_salient.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(
            channel_average + channel_salient
        )
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE