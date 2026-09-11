MECHANISM: Final binary search of the geometric-fusion ranking boundary

HYPOTHESIS: An 11/128 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.

INTENDED_EDIT: Increase geometric fusion from 1/64 to 11/128 while retaining the trained model, established views, view weights, and compensated calibration.

EVIDENCE: A 5/64 blend preserved 9,257 correct with 0.21436964797973632 cross-entropy, while 3/32 lost one correct but reduced cross-entropy further; 11/128 is their exact midpoint and isolates the remaining ranking boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(63.0 / 64.0),
            geometric_log_probs + math.log(1.0 / 64.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(117.0 / 128.0),
            geometric_log_probs + math.log(11.0 / 128.0),
        )
>>>>>>> REPLACE