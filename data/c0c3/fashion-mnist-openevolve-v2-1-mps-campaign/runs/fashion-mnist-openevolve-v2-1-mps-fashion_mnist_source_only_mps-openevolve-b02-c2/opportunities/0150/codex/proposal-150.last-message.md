MECHANISM: Interleaved-lattice four-peak channel saliency

HYPOTHESIS: Averaging maxima from four interleaved spatial lattices will exceed 9,322 correct predictions by approximating successful global top-four saliency without ranking cost or forced regional coverage, while completing within the reliable runtime envelope.

INTENDED_EDIT: Replace each channel’s hard global maximum with the mean maximum across the four checkerboard-parity lattices; leave optimization, training, and confidence-adaptive flip fusion unchanged.

EVIDENCE: Global top-four saliency produced the best verified count of 9,322 but repeatedly timed out, whereas hard maximum reliably reached 9,320. Regional and local pooling regressed because they constrain where salient activations must occur; interleaved lattices let all four peaks come from anywhere while avoiding `topk`.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        lattice_maxima = torch.stack(
            (
                features[:, :, 0::2, 0::2].amax(dim=(-2, -1)),
                features[:, :, 0::2, 1::2].amax(dim=(-2, -1)),
                features[:, :, 1::2, 0::2].amax(dim=(-2, -1)),
                features[:, :, 1::2, 1::2].amax(dim=(-2, -1)),
            ),
            dim=2,
        )
        channel_salient = lattice_maxima.mean(dim=2, keepdim=True)
        channel_salient = channel_salient.transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE