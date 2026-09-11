MECHANISM: Near-ceiling classifier bottleneck widening

HYPOTHESIS: Widening the fused-feature MLP from 48 to 56 units will exceed 9,328 correct predictions by increasing nonlinear classification capacity while retaining the successful architecture and training schedule.

INTENDED_EDIT: Increase the classifier hidden width to 56, raising learned parameters from 224,442 to 249,618—just below the 250,000 limit.

EVIDENCE: The nonlinear fused-feature classifier previously outperformed the standalone spatial readout (9,316 versus 9,307 correct), motivating allocation of the remaining parameter budget to the stronger MLP path.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 56),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(56, 10),
        )
>>>>>>> REPLACE