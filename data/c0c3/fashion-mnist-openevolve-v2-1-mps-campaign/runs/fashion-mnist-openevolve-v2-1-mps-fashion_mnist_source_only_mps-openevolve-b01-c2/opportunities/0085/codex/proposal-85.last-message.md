MECHANISM: Zero-initialized residual refinement

HYPOTHESIS: Initializing the refinement branch’s final BatchNorm scale to zero will stabilize early optimization and increase validation_correct from 9,330 to at least 9,331 without increasing parameters or runtime.

INTENDED_EDIT: Start the residual refinement block as an identity mapping while allowing it to learn progressively during the fixed training budget.

EVIDENCE: The best verified design reaches 9,330 correct, while added-computation architecture, augmentation, consistency, and EMA experiments repeatedly timed out; this standard residual initialization changes neither computation nor parameter count.

<<<<<<< SEARCH
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
=======
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        nn.init.zeros_(self.refine[-1].weight)
        self.classifier = nn.Sequential(
>>>>>>> REPLACE