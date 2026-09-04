MECHANISM: Per-channel affine calibration of shared channel attention

HYPOTHESIS: Restoring hard-maximum attention and adding identity-initialized per-channel scale and bias will exceed 9,322 correct predictions by preserving the successful shared descriptor kernel while allowing lightweight channel-specific calibration.

INTENDED_EDIT: Restore the reliable global-average-plus-maximum descriptor and affine-calibrate its shared attention output with 128 learned parameters, keeping the model at 249,983 parameters.

EVIDENCE: Shared average-plus-maximum attention reached 9,320 correct, whereas giving the two descriptors separate kernels fell to 9,300; this preserves beneficial kernel sharing and tests channel-specific flexibility along an orthogonal dimension.

<<<<<<< SEARCH
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
=======
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.channel_scale = nn.Parameter(torch.ones(1, 1, 64))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, 64))
        self.spatial_attention = nn.Conv2d(
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_salient = F.avg_pool2d(features, kernel_size=2, stride=1)
        channel_salient = F.adaptive_max_pool2d(channel_salient, 1)
        channel_salient = channel_salient.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
        channel_gate = channel_gate * self.channel_scale + self.channel_bias
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE