MECHANISM: Compute-light dense-head widening

HYPOTHESIS: Widening the classifier hidden layer from 88 to 92 units will raise validation_correct above 9,260 while staying within the runtime and 250,000-parameter limits.

INTENDED_EDIT: Use the remaining parameter budget to increase dense-head capacity, producing a 249,086-parameter model without adding convolutional or evaluation-time computation.

EVIDENCE: Both convolutional refinement attempts timed out, whereas the 239,634-parameter incumbent completes reliably; adding four dense units costs only 9,452 parameters and negligible computation compared with another convolutional block.

<<<<<<< SEARCH
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
=======
            nn.Linear(48 * 7 * 7, 92),
            nn.GELU(),
            nn.Linear(92, 10),
>>>>>>> REPLACE