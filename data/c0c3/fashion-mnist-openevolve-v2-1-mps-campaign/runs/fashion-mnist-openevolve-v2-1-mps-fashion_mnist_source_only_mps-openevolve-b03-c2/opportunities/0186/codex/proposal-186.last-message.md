MECHANISM: Binary search of the geometric-fusion ranking boundary

HYPOTHESIS: A 5/64 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214378804397583.

INTENDED_EDIT: Increase the geometric fusion weight from 1/32 to 5/64, retaining the trained model, established views, view weights, and compensated calibration.

EVIDENCE: The 1/16 blend retained 9,257 correct with 0.214378804397583 cross-entropy, while 3/32 lost one correct but further reduced cross-entropy; 5/64 is their midpoint and directly probes the remaining ranking boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(31.0 / 32.0),
            geometric_log_probs + math.log(1.0 / 32.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(59.0 / 64.0),
            geometric_log_probs + math.log(5.0 / 64.0),
        )
>>>>>>> REPLACE