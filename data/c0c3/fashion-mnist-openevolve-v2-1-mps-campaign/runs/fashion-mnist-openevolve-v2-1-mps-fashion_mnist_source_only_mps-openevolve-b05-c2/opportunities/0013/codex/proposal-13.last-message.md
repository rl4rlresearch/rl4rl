MECHANISM: Accuracy-boundary diagonal-view bisection

HYPOTHESIS: A diagonal weight of 0.71875 will retain 9,192 correct predictions while lowering cross-entropy below the 0.2374336 achieved at 0.725.

INTENDED_EDIT: Reduce diagonal translation-view weight from 0.725 to 0.71875 and adjust ensemble normalization from 15.8 to 15.75.

EVIDENCE: Weight 0.725 achieved 9,192 correct, while 0.7125 achieved 9,191 correct with lower cross-entropy; bisecting this interval is the most direct test for the lowest weight that preserves the leading correct count.

<<<<<<< SEARCH
                weight = 0.725 if is_diagonal else 1.0
=======
                weight = 0.71875 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.8
=======
        return logit_sum / 15.75
>>>>>>> REPLACE