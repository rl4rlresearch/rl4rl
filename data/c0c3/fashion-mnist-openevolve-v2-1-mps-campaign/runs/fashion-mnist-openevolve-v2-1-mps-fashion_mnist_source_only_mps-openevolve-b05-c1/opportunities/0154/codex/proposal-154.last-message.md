MECHANISM: Channel-preserving spatial compression with a widened nonlinear head

HYPOTHESIS: Replacing the 30-unit full-resolution bottleneck with 4×4 adaptive pooling and a 91-unit head will exceed 9,360 correct predictions by retaining all 64 learned channels while increasing nonlinear decision capacity within the parameter ceiling.

INTENDED_EDIT: Pool each 64-channel feature map from 7×7 to 4×4 before flattening, then widen the classifier hidden layer from 30 to 91.

EVIDENCE: The 64→20 channel-projection head fell to 9,326 correct, suggesting that discarding channel semantics was harmful; this patch instead obtains comparable head capacity by compressing spatial resolution while preserving every backbone channel.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 91),
            nn.BatchNorm1d(91),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(91, 10),
        )
>>>>>>> REPLACE