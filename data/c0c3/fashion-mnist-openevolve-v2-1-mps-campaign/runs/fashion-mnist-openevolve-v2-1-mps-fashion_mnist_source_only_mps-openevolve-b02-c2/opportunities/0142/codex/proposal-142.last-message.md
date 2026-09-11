MECHANISM: Attenuated peak channel attention

HYPOTHESIS: Weighting hard-max evidence by 0.85 will exceed 9,322 correct predictions by reducing single-activation dominance like successful top-four saliency, while the fused bias-free attention call preserves the reliable runtime envelope.

INTENDED_EDIT: Fuse the shared mean/max channel-attention calls and modestly downweight the maximum descriptor.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322 but often exceeded the time limit, while hard maxima reliably finished with 9,320; the eighth-power substitute regressed, motivating a cheaper direct adjustment of peak influence.

<<<<<<< SEARCH
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_gate = self.channel_attention(
            channel_average + 0.85 * channel_maximum
        )
>>>>>>> REPLACE