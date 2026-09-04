MECHANISM: Compute-efficient higher-update training

HYPOTHESIS: Replacing GELU with cheaper in-place ReLU activations will let batch size 28 complete within the time limit and exceed 9,254 correct predictions through approximately 446 additional optimizer updates.

INTENDED_EDIT: Reduce batch size from 32 to 28 and replace all four GELU activations with in-place ReLU, preserving the architecture, optimizer, loss, schedule, and strided EMA.

EVIDENCE: Batch-size reductions from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250, but batch 24 timed out; reducing activation cost targets that runtime barrier while retaining the proven higher-update mechanism.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 28
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.ReLU(inplace=True),
            nn.Linear(88, 10),
        )
>>>>>>> REPLACE