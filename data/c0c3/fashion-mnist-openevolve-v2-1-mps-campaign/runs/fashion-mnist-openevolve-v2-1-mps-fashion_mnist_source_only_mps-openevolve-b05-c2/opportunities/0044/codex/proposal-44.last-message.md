MECHANISM: Continued accuracy-invariant logit sharpening

HYPOTHESIS: Dividing the best-known ensemble logits by 98.0% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23623599700927733.

INTENDED_EDIT: Restore the best-performing diagonal TTA weight and increase post-hoc sharpening from 1.0% to 2.0%.

EVIDENCE: Sharpening by 0.5% and 1.0% successively reduced cross-entropy to 0.23682696762084962 and 0.23623599700927733 while retaining all 9,192 predictions; positive global logit scaling preserves class ordering.

<<<<<<< SEARCH
                weight = 0.71613826751708984375 if is_diagonal else 1.0
=======
                weight = 0.7161376953125 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.72910614013671875
=======
        return logit_sum / 15.41451953125
>>>>>>> REPLACE