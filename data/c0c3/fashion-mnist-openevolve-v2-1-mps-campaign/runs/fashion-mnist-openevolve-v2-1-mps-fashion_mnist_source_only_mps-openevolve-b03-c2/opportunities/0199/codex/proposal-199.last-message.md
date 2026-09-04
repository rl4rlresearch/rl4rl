MECHANISM: Terminal geometric-fusion boundary bisection

HYPOTHESIS: A 41833/524288 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870040893555.

INTENDED_EDIT: Set the arithmetic–geometric probability blend to the exact midpoint between the best qualified 5229/65536 blend and the nearest accuracy-reducing 20917/262144 blend, retaining all other settings.

EVIDENCE: The 5229/65536 blend achieved 9,257 correct with 0.21436870040893555 cross-entropy, while 20917/262144 lost one correct; 41833/524288 is the sole untested midpoint between them.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(7539.0 / 8192.0),
            geometric_log_probs + math.log(653.0 / 8192.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(482455.0 / 524288.0),
            geometric_log_probs + math.log(41833.0 / 524288.0),
        )
>>>>>>> REPLACE