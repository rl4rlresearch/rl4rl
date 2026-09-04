MECHANISM: Faster activations fund higher-update optimization

HYPOTHESIS: Replacing GELU with in-place ReLU will make batch-size-44 training finish within the runtime limit and exceed 9,280 correct predictions through approximately 9% more optimizer steps.

INTENDED_EDIT: Use batch size 44 and replace all feature and classifier GELU activations with lower-cost in-place ReLU activations; preserve augmentation, optimizer, schedule, capacity, and validation ensemble.

EVIDENCE: Batch 48 reached 9,280 correct after outperforming batch 64 through additional updates, while batch 44 timed out with the architecture unchanged; reducing activation overhead directly targets that runtime failure while retaining the proposed higher-update regime.

<<<<<<< SEARCH
BATCH_SIZE = 48
=======
BATCH_SIZE = 44
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
=======
            nn.Linear(96 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
>>>>>>> REPLACE