MECHANISM: Budget-maximized fixed spatial bottleneck

HYPOTHESIS: Increasing the fixed-position bottleneck from 30 to 31 features will exceed 9,249 correct predictions by adding one spatial template while retaining the location-sensitive head that outperformed attention pooling.

INTENDED_EDIT: Use the remaining parameter budget to widen the classifier bottleneck by one feature, increasing parameters from 245,044 to 248,191 without materially changing runtime.

EVIDENCE: Replacing the 30-feature fixed-position head with content-addressed attention pooling reduced correctness to 9,228, indicating that fixed spatial summaries are valuable; a controlled one-feature widening tests additional capacity within that successful representation.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 31),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(31, 10),
        )
>>>>>>> REPLACE