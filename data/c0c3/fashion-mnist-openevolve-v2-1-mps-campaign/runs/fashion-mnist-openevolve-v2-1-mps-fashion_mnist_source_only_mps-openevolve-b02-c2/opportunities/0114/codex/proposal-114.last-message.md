MECHANISM: Differentiable global upper-tail channel saliency

HYPOTHESIS: Softmax-weighted global saliency will exceed 9,322 correct predictions by approximating top-four pooling with dense gradients while avoiding ranking overhead.

INTENDED_EDIT: Replace hard channel maxima with a topology-independent softmax-weighted average of strong spatial activations.

EVIDENCE: Exact top-four pooling achieved the best verified count of 9,322 versus 9,320 for hard maxima, while contiguous 2×2 smoothing regressed to 9,297 and ranking-based variants repeatedly timed out.

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
        flattened_features = features.flatten(2)
        saliency_weights = F.softmax(flattened_features, dim=2)
        channel_salient = (
            flattened_features * saliency_weights
        ).sum(dim=2, keepdim=True).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE