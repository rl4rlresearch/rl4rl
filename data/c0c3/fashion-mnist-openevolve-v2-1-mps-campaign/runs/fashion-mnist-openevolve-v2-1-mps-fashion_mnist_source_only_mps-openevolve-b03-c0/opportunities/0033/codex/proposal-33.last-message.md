MECHANISM: Learned early strided downsampling

HYPOTHESIS: Replacing the first max pool with a learned stride-2 convolution while retaining a 41-unit classifier will exceed 9,319 correct predictions by preserving fine local structure during the first resolution reduction.

INTENDED_EDIT: Use a normalized, activated 3×3 stride-2 convolution for the first downsampling stage and reduce the classifier width from 44 to 41, yielding 249,773 learned parameters.

EVIDENCE: The full 7×7 refinement convolution reached 9,315 correct and improved cross-entropy to 0.21767, indicating learned spatial processing is competitive; placing that capacity at the earlier downsampling boundary preserves more classifier capacity and directly tests learned rather than fixed pooling.

<<<<<<< SEARCH
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
=======
            ResidualBlock(32),
            nn.Conv2d(
                32, 32, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 44),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(44, 10),
=======
            nn.Linear(64 * 7 * 7, 41),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(41, 10),
>>>>>>> REPLACE