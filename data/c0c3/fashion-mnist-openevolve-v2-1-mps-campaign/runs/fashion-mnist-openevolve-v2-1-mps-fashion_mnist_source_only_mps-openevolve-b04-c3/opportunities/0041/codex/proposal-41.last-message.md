MECHANISM: Coarse spatial pooling with a wider nonlinear head

HYPOTHESIS: Replacing the fixed 4,704-to-38 bottleneck with 4×4 adaptive average pooling and a 112-unit head will exceed 9,240 correct predictions by trading redundant pixel-level positions for greater classifier rank without increasing runtime.

INTENDED_EDIT: Pool the 7×7 feature maps to 4×4 and widen the classifier to 112 units, reducing the model from 245,040 to approximately 239,282 parameters while preserving all verified training and inference settings.

EVIDENCE: The 245,040-parameter flat-head design achieved 9,240 correct, whereas merely widening its bottleneck to 39 fell to 9,223 and more computationally involved spatial alternatives timed out; this motivates a low-cost reallocation of dense capacity rather than adding parameters or convolutional work.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 112),
            nn.LayerNorm(112),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(112, 10),
        )
>>>>>>> REPLACE