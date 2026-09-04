MECHANISM: Final geometric-fusion boundary bisection

HYPOTHESIS: A 1307/16384 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436874084472657.

INTENDED_EDIT: Increase geometric fusion to the exact midpoint between the best qualified 653/8192 blend and the nearest accuracy-reducing 327/4096 blend, retaining all other settings.

EVIDENCE: The 653/8192 blend retained 9,257 correct at 0.21436874084472657 cross-entropy, while 327/4096 lost one correct; 1307/16384 bisects the remaining boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(943.0 / 1024.0),
            geometric_log_probs + math.log(81.0 / 1024.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(15077.0 / 16384.0),
            geometric_log_probs + math.log(1307.0 / 16384.0),
        )
>>>>>>> REPLACE