MECHANISM: Small-gamma residual initialization

HYPOTHESIS: Initializing both residual branches at 0.1 strength will exceed 9,328 correct predictions by improving early optimization stability while preserving gradient flow, architecture, inference, and runtime.

INTENDED_EDIT: Initialize the final BatchNorm scale in each existing residual branch to 0.1 instead of the default 1.0.

EVIDENCE: Dense-head widening reached only 9,300, auxiliary supervision reached 9,294, and added spatial refinement timed out; this tests optimization of the verified 224,442-parameter pathway without adding computation or changing its inference fusion.

<<<<<<< SEARCH
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
=======
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        with torch.no_grad():
            self.residual1[1].weight.fill_(0.1)
            self.residual2[1].weight.fill_(0.1)
        self.pool = nn.MaxPool2d(2)
>>>>>>> REPLACE