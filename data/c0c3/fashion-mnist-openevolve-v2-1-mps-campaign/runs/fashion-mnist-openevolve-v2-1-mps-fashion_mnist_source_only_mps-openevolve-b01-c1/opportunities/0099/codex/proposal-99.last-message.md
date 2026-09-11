MECHANISM: Conservative residual-width reallocation

HYPOTHESIS: Widening only the residual output from 56 to 60 channels while reducing the classifier from 64 to 60 units will exceed 9,257 correct predictions by adding spatial feature capacity without the over-reallocation of the unsuccessful 64-channel design.

INTENDED_EDIT: Change the residual and shortcut output width to 60 and the flattened classifier width to 60, yielding 239,966 learned parameters.

EVIDENCE: Reallocating capacity toward convolutional features improved correctness from 9,247 to 9,257, but the more aggressive 44→64-channel design fell to 9,238; this tests a conservative intermediate allocation while retaining the proven 40-channel second stage.

<<<<<<< SEARCH
        self.residual = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=1, bias=False),
            nn.BatchNorm2d(56),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56 * 7 * 7, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 10),
        )
=======
        self.residual = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(60),
            nn.GELU(),
            nn.Conv2d(60, 60, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(60),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=1, bias=False),
            nn.BatchNorm2d(60),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60 * 7 * 7, 60),
            nn.GELU(),
            nn.LayerNorm(60),
            nn.Linear(60, 10),
        )
>>>>>>> REPLACE