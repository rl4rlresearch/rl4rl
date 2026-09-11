MECHANISM: Channel-preserving spatial compression with a widened nonlinear head

HYPOTHESIS: Adaptive 4×4 pooling with a 93-unit hidden layer will exceed 9,360 correct predictions by retaining all 64 channel identities while tripling nonlinear head capacity within the parameter ceiling.

INTENDED_EDIT: Insert adaptive average pooling before flattening and widen the classifier bottleneck from 30 to 93 units, producing 249,517 learned parameters.

EVIDENCE: The 64→20 projection head fell to 9,326 correct, suggesting that discarding channel semantics is harmful; the prior channel-preserving pooling proposal could not be verified, so this capacity-maximized version remains an informative untested alternative.

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
            nn.Linear(64 * 4 * 4, 93),
            nn.BatchNorm1d(93),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(93, 10),
        )
>>>>>>> REPLACE