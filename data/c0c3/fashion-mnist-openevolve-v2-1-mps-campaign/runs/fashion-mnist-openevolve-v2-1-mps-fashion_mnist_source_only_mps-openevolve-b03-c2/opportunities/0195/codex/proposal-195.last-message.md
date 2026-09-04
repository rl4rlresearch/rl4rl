MECHANISM: Terminal geometric-fusion boundary bisection

HYPOTHESIS: A 2615/32768 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870956420898.

INTENDED_EDIT: Increase geometric fusion from 1307/16384 to 2615/32768, the sole midpoint between the best qualified blend and the nearest accuracy-reducing blend, while retaining all other settings.

EVIDENCE: The 1307/16384 blend retained 9,257 correct at 0.21436870956420898 cross-entropy, while 327/4096 lost one correct; 2615/32768 exactly bisects that final unresolved interval.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(59.0 / 64.0),
            geometric_log_probs + math.log(5.0 / 64.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(30153.0 / 32768.0),
            geometric_log_probs + math.log(2615.0 / 32768.0),
        )
>>>>>>> REPLACE