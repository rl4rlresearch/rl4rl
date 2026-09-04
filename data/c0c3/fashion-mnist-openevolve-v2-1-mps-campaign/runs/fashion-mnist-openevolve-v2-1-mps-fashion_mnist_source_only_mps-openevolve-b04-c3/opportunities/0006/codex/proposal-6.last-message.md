MECHANISM: Additional mid-resolution residual refinement

HYPOTHESIS: Adding a second 48-channel residual depthwise block while retaining batch size 64 will exceed 9,229 correct predictions by using the remaining parameter headroom for deeper feature refinement.

INTENDED_EDIT: Add one residual depthwise block at the 14×14 feature stage, increasing parameters from 245,040 to approximately 247,968 without changing the qualified optimizer, augmentation, or batch size.

EVIDENCE: The current batch-64 residual design achieved 9,229 correct with 245,040 parameters, while translation augmentation reduced correctness to 9,058; this motivates preserving its established training procedure and testing a focused capacity increase below the 250,000-parameter ceiling.

<<<<<<< SEARCH
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
=======
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
>>>>>>> REPLACE