MECHANISM: Accuracy-boundary diagonal-view shrinkage

HYPOTHESIS: A diagonal weight of 0.725 will retain the 9,192 correct predictions achieved at 0.75 while lowering cross-entropy by moving halfway toward the better-calibrated 0.70 ensemble.

INTENDED_EDIT: Increase diagonal translation-view weight from 0.70 to 0.725 and adjust ensemble normalization from 15.6 to 15.8.

EVIDENCE: Weight 0.75 produced 9,192 correct at 0.237446 cross-entropy, whereas 0.70 produced 9,191 correct at a lower 0.237421; bisecting this narrow accuracy boundary is the most informative refinement.

<<<<<<< SEARCH
                weight = 0.70 if is_diagonal else 1.0
=======
                weight = 0.725 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.6
=======
        return logit_sum / 15.8
>>>>>>> REPLACE