MECHANISM: Descriptor-specific dual-statistic channel attention

HYPOTHESIS: Restoring the 9,320-correct architecture and giving global-average and global-maximum descriptors independent channel-attention kernels will exceed 9,320 correct predictions while remaining below 250,000 parameters.

INTENDED_EDIT: Remove the regressive depthwise-grouped residual block, restore the 48-unit classifier, and replace the shared attention kernel with separately learned, identity-initialized average and maximum kernels.

EVIDENCE: The shared dual-statistic design achieved 9,320 correct, while reallocating classifier capacity to the extra spatial block fell to 9,261; separate three-weight kernels preserve the successful signals while allowing their distinct statistics to be interpreted independently.

<<<<<<< SEARCH
        self.spatial_refine_conv = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, groups=64, bias=False
        )
        self.spatial_refine_norm = nn.BatchNorm2d(64)
        self.spatial_mix_conv = nn.Conv2d(
            64, 64, kernel_size=1, groups=2, bias=False
        )
        self.spatial_mix_norm = nn.BatchNorm2d(64)
        nn.init.zeros_(self.spatial_mix_norm.weight)

        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
=======
        self.channel_average_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.channel_maximum_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_average_attention.weight)
        nn.init.zeros_(self.channel_maximum_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 47),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(47, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual = features
        features = F.gelu(
            self.spatial_refine_norm(self.spatial_refine_conv(features))
        )
        features = self.spatial_mix_norm(self.spatial_mix_conv(features))
        features = features + residual

        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_average = F.adaptive_avg_pool2d(features, 1)
        channel_average = channel_average.flatten(2).transpose(1, 2)
        channel_maximum = F.adaptive_max_pool2d(features, 1)
        channel_maximum = channel_maximum.flatten(2).transpose(1, 2)
        channel_gate = self.channel_average_attention(channel_average)
        channel_gate = channel_gate + self.channel_maximum_attention(
            channel_maximum
        )
>>>>>>> REPLACE