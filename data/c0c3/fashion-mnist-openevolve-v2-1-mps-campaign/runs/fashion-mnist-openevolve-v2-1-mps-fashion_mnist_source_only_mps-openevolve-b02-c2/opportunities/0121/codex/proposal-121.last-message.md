MECHANISM: Descriptor-contrast residual channel attention

HYPOTHESIS: Preserving exact top-four saliency while learning a three-parameter contrast kernel between average and salient evidence will exceed 9,322 correct predictions by relaxing the shared-kernel constraint without changing initialization or material runtime.

INTENDED_EDIT: Add a zero-initialized residual kernel that specializes channel attention according to the difference between top-four and average descriptors, while fusing their existing shared response.

EVIDENCE: Exact top-four saliency produced the best verified count of 9,322, whereas alternative saliency approximations regressed; this patch preserves that descriptor and makes only its fusion more expressive.

<<<<<<< SEARCH
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.spatial_attention = nn.Conv2d(
=======
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        self.channel_attention_contrast = nn.Parameter(torch.zeros(1, 1, 3))
        self.spatial_attention = nn.Conv2d(
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_gate = self.channel_attention(channel_average)
        channel_gate = channel_gate + self.channel_attention(channel_salient)
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
=======
        channel_gate = self.channel_attention(
            channel_average + channel_salient
        )
        channel_gate = channel_gate + F.conv1d(
            channel_salient - channel_average,
            self.channel_attention_contrast,
            padding=1,
        )
        channel_gate = channel_gate.transpose(1, 2).unsqueeze(-1)
>>>>>>> REPLACE