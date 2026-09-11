MECHANISM: Eighth-power upper-tail channel saliency

HYPOTHESIS: Replacing hard channel maxima with an eighth-power mean will finish within the reliable pooling runtime envelope and exceed 9,322 correct predictions by approximating top-four evidence with dense gradients and reduced single-outlier sensitivity.

INTENDED_EDIT: Replace global-maximum channel evidence with a parameter-free eighth-power mean while preserving the qualified architecture, training objective, schedule, and probability-space flip ensemble.

EVIDENCE: Exact top-four saliency achieved the best verified count of 9,322 but repeatedly risked timeout, while hard maxima reliably reached 9,320. Unlike the regressing peak-2×2 approximation, a power mean retains global upper-tail evidence without ranking or imposing spatial locality.

<<<<<<< SEARCH
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_salient = features.square().square().square()
        channel_salient = channel_salient.mean(dim=(2, 3), keepdim=True)
        channel_salient = channel_salient.sqrt().sqrt().sqrt()
        channel_salient = channel_salient.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
>>>>>>> REPLACE