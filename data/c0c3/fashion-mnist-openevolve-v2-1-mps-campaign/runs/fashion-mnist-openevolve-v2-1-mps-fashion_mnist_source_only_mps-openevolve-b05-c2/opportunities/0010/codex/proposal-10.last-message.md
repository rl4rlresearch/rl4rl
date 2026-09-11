MECHANISM: Local diagonal-view weight refinement

HYPOTHESIS: Increasing the diagonal-view weight from 0.75 to 0.80 will retain the 9,192 correct predictions and may add corrections from diagonal shifts while remaining less noisy than uniform weighting.

INTENDED_EDIT: Use 0.80-weighted diagonal translation views and update the ensemble normalization from 16.0 to 16.4.

EVIDENCE: Diagonal weights 0.50, 0.70, 0.75, and 1.00 yielded 9,190, 9,191, 9,192, and 9,191 correct respectively, making a small upward refinement from the best observed weight the most informative next test.

<<<<<<< SEARCH
                weight = 0.75 if is_diagonal else 1.0
=======
                weight = 0.80 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 16.0
=======
        return logit_sum / 16.4
>>>>>>> REPLACE