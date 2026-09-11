MECHANISM: Descriptor-specific channel-attention kernels

HYPOTHESIS: Giving global-average and global-maximum descriptors independent three-tap kernels will exceed 9,322 correct predictions by allowing complementary channel statistics to learn distinct interactions without changing the proven saliency computation.

INTENDED_EDIT: Add a separately initialized three-parameter kernel for maximum-channel evidence while preserving paired-view training, flip ensembling, and all other behavior.

EVIDENCE: Hard-maximum attention reliably achieved 9,320 correct, while alternative saliency reductions produced only a two-correct gain or regressed; preserving maximum evidence and relaxing its shared-kernel constraint is a low-cost, previously untested refinement.

<<<<<<< SEARCH
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
=======
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.channel_max_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )
        nn.init.zeros_(self.channel_attention.weight)
        nn.init.zeros_(self.channel_max_attention.weight)
        nn.init.zeros_(self.spatial_attention.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_maximum)
=======
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_max_attention(channel_maximum)
>>>>>>> REPLACE