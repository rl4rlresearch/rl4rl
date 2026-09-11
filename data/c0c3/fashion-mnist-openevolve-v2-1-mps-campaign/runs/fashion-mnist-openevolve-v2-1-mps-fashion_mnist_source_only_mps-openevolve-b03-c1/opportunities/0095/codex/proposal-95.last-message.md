MECHANISM: Compute-neutral factorized dense classifier

HYPOTHESIS: Replacing the 160-unit classifier layer with two 138-unit nonlinear layers will exceed 9,166 correct predictions by increasing decision depth while retaining the successful flattened representation, dropout strengths, and nearly identical parameter and compute budgets.

INTENDED_EDIT: Reallocate the dense head into two GELU stages, reducing the model by 68 parameters to 249,721 without adding meaningful runtime.

EVIDENCE: Filter, loss, and TTA refinements failed to surpass 9,166, while the spatial-attention replacement timed out and reduced dropout fell to 9,141; this motivates a lightweight head-capacity change that preserves the established representation and regularization.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 138),
            nn.GELU(),
            nn.Linear(138, 138),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(138, 10),
        )
>>>>>>> REPLACE