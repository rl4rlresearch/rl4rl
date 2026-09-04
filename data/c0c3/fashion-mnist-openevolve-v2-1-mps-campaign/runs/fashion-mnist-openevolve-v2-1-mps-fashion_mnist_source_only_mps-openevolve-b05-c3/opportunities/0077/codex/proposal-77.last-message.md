MECHANISM: Zero-initialized residual refinement

HYPOTHESIS: Initializing each refinement BatchNorm scale to zero will exceed 9,193 correct predictions by starting every stage as a stable single-convolution path while allowing residual refinements to emerge during training.

INTENDED_EDIT: Zero-initialize the learned scale of each stage’s second BatchNorm without changing architecture, parameter count, runtime, curriculum, EMA, or TTA.

EVIDENCE: Residual stages improved the best completed result from 9,172 to 9,193 correct, supporting feature preservation and gradient flow; zero-initialization strengthens that successful identity bias specifically at initialization.

<<<<<<< SEARCH
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
=======
        self.norm2 = nn.BatchNorm2d(out_channels)
        nn.init.zeros_(self.norm2.weight)
        self.pool = nn.MaxPool2d(2)
>>>>>>> REPLACE