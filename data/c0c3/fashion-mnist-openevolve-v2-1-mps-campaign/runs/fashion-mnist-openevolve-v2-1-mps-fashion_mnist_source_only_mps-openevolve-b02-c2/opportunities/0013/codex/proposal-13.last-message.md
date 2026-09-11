MECHANISM: Tri-statistic channel attention

HYPOTHESIS: Adding global channel dispersion to the verified average-plus-maximum attention will exceed 9,320 correct predictions without increasing learned parameters.

INTENDED_EDIT: Preserve the best architecture and training regimen while adding per-channel spatial standard deviation to the shared channel-attention gate.

EVIDENCE: Replacing average-only attention with shared average-plus-maximum evidence improved paired-view validation from 9,286 to 9,320 correct; testing a third complementary statistic is the most isolated extension of that gain.

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
        channel_deviation = (
            F.adaptive_avg_pool2d(features.square(), 1)
            - channel_average.square()
        ).clamp_min(1e-6).sqrt()
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_deviation = channel_deviation.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate + self.channel_attention(channel_deviation)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE