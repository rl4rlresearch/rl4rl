MECHANISM: Two-color checkerboard peak attention

HYPOTHESIS: Averaging maxima from two interleaved checkerboard lattices will finish within the reliable runtime envelope and exceed 9,322 correct predictions by providing two robust salient gradients without `topk` or four separate lattice reductions.

INTENDED_EDIT: Replace single hard-max channel evidence with the mean maxima of the two flattened checkerboard parities, and fuse the shared average/salient attention computation into one convolution call.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly timed out, while hard-max attention reliably finished near 9,320; the four-lattice approximation also timed out, motivating a cheaper two-lattice compromise.

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
        flattened_features = features.flatten(2)
        channel_salient = 0.5 * (
            flattened_features[..., 0::2].amax(dim=2)
            + flattened_features[..., 1::2].amax(dim=2)
        )
        channel_salient = channel_salient.unsqueeze(1)
        channel_gates = self.channel_attention(
            torch.cat((channel_average, channel_salient), dim=0)
        )
        average_gate, salient_gate = channel_gates.chunk(2, dim=0)
        channel_gate = average_gate + salient_gate
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE