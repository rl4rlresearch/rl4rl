MECHANISM: BatchNorm–ReLU spatial feature extraction

HYPOTHESIS: Replacing the three spatial GELUs with ReLUs will finish more reliably while retaining or exceeding 9,258 correct predictions through faster, canonical BatchNorm-activated convolution blocks.

INTENDED_EDIT: Use ReLU activations in the convolutional feature extractor while preserving the architecture, classifier head, optimizer, EMA, and label smoothing.

EVIDENCE: The current design reached 9,258 correct but required 80.92 seconds, while even runtime-neutral variants repeatedly timed out; ReLU reduces activation overhead without narrowing features or changing parameter count, unlike the timed-out depthwise-separable redesign.

<<<<<<< SEARCH
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
=======
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE