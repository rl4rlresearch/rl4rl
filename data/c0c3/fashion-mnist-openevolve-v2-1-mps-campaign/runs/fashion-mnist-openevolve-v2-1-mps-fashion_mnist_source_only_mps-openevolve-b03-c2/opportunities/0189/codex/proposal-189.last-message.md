MECHANISM: Sub-percent geometric-fusion boundary search

HYPOTHESIS: A 41/512 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.

INTENDED_EDIT: Increase geometric fusion from 5/64 to 41/512, the midpoint between the best qualified blend and the nearest tested accuracy-reducing blend.

EVIDENCE: The 5/64 blend retained 9,257 correct at 0.21436964797973632 cross-entropy, while 21/256 lost one correct but lowered cross-entropy; 41/512 bisects this unresolved ranking boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(59.0 / 64.0),
            geometric_log_probs + math.log(5.0 / 64.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(471.0 / 512.0),
            geometric_log_probs + math.log(41.0 / 512.0),
        )
>>>>>>> REPLACE