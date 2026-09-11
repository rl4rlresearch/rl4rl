MECHANISM: Pre-classifier spatial feature normalization

HYPOTHESIS: Adding channel-wise BatchNorm after the convolutional trunk will exceed 9,314 correct predictions by stabilizing the flattened classifier input while remaining below the parameter ceiling.

INTENDED_EDIT: Append a 64-channel BatchNorm2d layer to the feature extractor, adding 128 learned parameters without otherwise changing training or inference.

EVIDENCE: BatchNorm on the 30-feature bottleneck improved correctness from 9,249 to 9,314; this applies the same successful conditioning mechanism at the remaining unnormalized trunk-to-classifier boundary.

<<<<<<< SEARCH
            ResidualBlock(64, 64),
        )
=======
            ResidualBlock(64, 64),
            nn.BatchNorm2d(64),
        )
>>>>>>> REPLACE