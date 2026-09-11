MECHANISM: Four-channel grouped spatial residual mixing

HYPOTHESIS: Using four-channel grouped spatial filters with the verified 12.5% learning-rate floor will exceed 9,240 correct predictions by learning local cross-channel shape interactions that independent depthwise filters cannot represent.

INTENDED_EDIT: Replace each depthwise residual convolution with a grouped 3×3 convolution whose filters jointly process four channels, increasing the model to 248,928 parameters, and use the strongest verified 12.5%-floor cosine schedule.

EVIDENCE: Schedule-only refinement plateaued at 9,240 correct with the 12.5% floor and degraded at 20%, motivating a representational change. All qualified architectures assume spatial filtering occurs independently per channel; the proposed grouped filters challenge that assumption with modest compute and remain below the parameter ceiling.

<<<<<<< SEARCH
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
=======
        self.grouped_spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels // 4,
            bias=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = F.gelu(self.norm1(self.depthwise(inputs)))
=======
        hidden = F.gelu(self.norm1(self.grouped_spatial(inputs)))
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE