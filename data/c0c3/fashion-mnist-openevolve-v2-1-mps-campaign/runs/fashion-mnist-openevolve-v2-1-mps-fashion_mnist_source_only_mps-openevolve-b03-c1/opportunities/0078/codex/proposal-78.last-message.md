MECHANISM: Reduced dense-head dropout regularization

HYPOTHESIS: Lowering classifier dropout from 15%/10% to 10%/5% will exceed 9,162 correct predictions by improving head adaptation within the fixed 1,564-step budget while retaining augmentation, label smoothing, weight decay, and EMA regularization.

INTENDED_EDIT: Reduce both classifier dropout rates without changing architecture, parameter count, evaluation, or runtime.

EVIDENCE: Early weak mixup reduced correctness from 9,162 to 9,130, indicating that additional regularization is harmful under the fixed exposure budget; modestly relaxing the model’s existing head regularization is the most direct complementary test.

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
            nn.Dropout(0.10),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(160, 10),
        )
>>>>>>> REPLACE