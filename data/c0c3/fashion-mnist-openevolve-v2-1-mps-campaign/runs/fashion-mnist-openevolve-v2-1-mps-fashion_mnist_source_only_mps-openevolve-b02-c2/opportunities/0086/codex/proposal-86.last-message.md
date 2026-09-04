MECHANISM: Four-site-normalized L4 channel saliency

HYPOTHESIS: Replacing hard maxima with a differentiable L4 upper-tail descriptor will exceed 9,322 correct predictions by capturing multiple strong activations like top-four pooling while avoiding its timeout-prone selection cost.

INTENDED_EDIT: Use positive L4 pooling normalized to four effective salient sites and fuse the two bias-free channel-attention calls into one linear operation.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322 versus 9,320 for hard maxima, while max–RMS pooling reached 9,317 and repeated top-k variants timed out; a higher-order pooled descriptor targets the successful upper tail more selectively than RMS without sorting.

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
        channel_salient = F.relu(features.flatten(2))
        channel_salient = torch.linalg.vector_norm(
            channel_salient, ord=4.0, dim=2, keepdim=True
        )
        channel_salient = channel_salient * (4.0 ** -0.25)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average + channel_salient)
>>>>>>> REPLACE