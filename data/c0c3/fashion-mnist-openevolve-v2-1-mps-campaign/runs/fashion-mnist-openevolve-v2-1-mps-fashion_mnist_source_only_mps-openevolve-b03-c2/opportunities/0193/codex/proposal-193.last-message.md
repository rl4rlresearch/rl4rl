MECHANISM: Geometric-fusion boundary bisection

HYPOTHESIS: A 653/8192 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436881332397462.

INTENDED_EDIT: Increase geometric fusion from 163/2048 to 653/8192, the exact midpoint between the best qualified blend and the nearest accuracy-reducing blend.

EVIDENCE: The 163/2048 blend retained 9,257 correct at 0.21436881332397462 cross-entropy, while 327/4096 lost one correct; 653/8192 bisects this remaining ranking boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(1885.0 / 2048.0),
            geometric_log_probs + math.log(163.0 / 2048.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(7539.0 / 8192.0),
            geometric_log_probs + math.log(653.0 / 8192.0),
        )
>>>>>>> REPLACE