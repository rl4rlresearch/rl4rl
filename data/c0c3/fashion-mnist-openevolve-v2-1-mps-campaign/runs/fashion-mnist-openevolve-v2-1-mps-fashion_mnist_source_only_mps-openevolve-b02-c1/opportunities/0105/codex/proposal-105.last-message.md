MECHANISM: Reduced dense-head regularization

HYPOTHESIS: Lowering classifier dropout from 0.15 to 0.10 will achieve at least 9,311 correct predictions by allowing the newly widened head to learn more fully during the fixed two-exposure budget without materially changing runtime.

INTENDED_EDIT: Reduce only the dropout probability in the verified 249,601-parameter normalized classifier head.

EVIDENCE: Widening and normalizing the dense head improved validation-correct from 9,286 to 9,310, indicating useful remaining head capacity; with strong translation, flip, BatchNorm, weight decay, and checkpoint-averaging regularization already present, slightly less dropout is a targeted test of underfitting.

<<<<<<< SEARCH
            nn.Dropout(0.15),
=======
            nn.Dropout(0.10),
>>>>>>> REPLACE