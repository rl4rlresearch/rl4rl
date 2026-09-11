MECHANISM: Hybrid ReLU–GELU feature extraction

HYPOTHESIS: Replacing only the highest-resolution GELU with ReLU will reduce training time while retaining at least 9,258 correct predictions because the deeper smooth activations remain unchanged.

INTENDED_EDIT: Use ReLU after the first convolution while preserving both deeper GELUs, the classifier, optimizer, EMA, and loss.

EVIDENCE: Replacing all three spatial GELUs with ReLUs reduced training time from 80.92 to 68.65 seconds but lost 23 correct predictions; changing only the most computationally expensive GELU tests whether most of that speedup is available without the accuracy loss caused by removing every smooth activation.

<<<<<<< SEARCH
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
>>>>>>> REPLACE