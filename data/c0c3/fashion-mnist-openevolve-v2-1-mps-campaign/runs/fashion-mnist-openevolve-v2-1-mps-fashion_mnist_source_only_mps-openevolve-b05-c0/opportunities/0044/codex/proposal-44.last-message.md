MECHANISM: Factorized spatial bottleneck with head-capacity reallocation

HYPOTHESIS: Replacing the costly 48→48 spatial convolution with a 24-channel bottleneck and widening the dense head will finish reliably while exceeding 9,254 correct predictions by preserving batch-32 optimization and comparable parameter capacity.

INTENDED_EDIT: Factor the third convolution into 48→24 pointwise reduction and 24→48 spatial expansion, then widen the classifier hidden layer from 88 to 92; estimated learned parameters remain below 250,000.

EVIDENCE: Batch-size reductions improved correct predictions from 9,210 to 9,235 to 9,250, favoring preservation of 3,126 updates, while the 75.95-second baseline and repeated subsequent timeouts motivate reducing per-example convolutional work.

<<<<<<< SEARCH
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(48, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
=======
            nn.Linear(48 * 7 * 7, 92),
            nn.GELU(),
            nn.Linear(92, 10),
>>>>>>> REPLACE