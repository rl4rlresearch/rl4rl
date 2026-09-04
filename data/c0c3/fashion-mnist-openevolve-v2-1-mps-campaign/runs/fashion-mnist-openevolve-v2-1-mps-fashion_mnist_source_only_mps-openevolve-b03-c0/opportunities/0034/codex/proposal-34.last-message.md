MECHANISM: Stabilized BatchNorm running statistics

HYPOTHESIS: Reducing BatchNorm momentum from 0.10 to 0.05 will exceed 9,319 correct predictions by reducing evaluation-time normalization noise without changing training dynamics, parameter count, or runtime materially.

INTENDED_EDIT: Set every BatchNorm layer’s running-statistics momentum to 0.05.

EVIDENCE: EMA weight consolidation remained competitive at 9,315 correct, suggesting endpoint stability matters; BatchNorm statistics are a distinct, untested source of endpoint noise that can be averaged more reliably at batch size 64.

<<<<<<< SEARCH
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
=======
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.05),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.05),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
=======
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.05),
            nn.SiLU(inplace=True),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.05),
>>>>>>> REPLACE