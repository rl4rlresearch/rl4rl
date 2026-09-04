MECHANISM: Sub-permille geometric-fusion boundary search

HYPOTHESIS: An 81/1024 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436964797973632.

INTENDED_EDIT: Increase geometric fusion from 5/64 to 81/1024, the exact midpoint between the best qualified blend and the nearest tested accuracy-reducing blend.

EVIDENCE: The 5/64 blend retained 9,257 correct at 0.21436964797973632 cross-entropy, while 41/512 lost one correct; 81/1024 bisects this remaining ranking boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(15.0 / 16.0),
            geometric_log_probs + math.log(1.0 / 16.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(943.0 / 1024.0),
            geometric_log_probs + math.log(81.0 / 1024.0),
        )
>>>>>>> REPLACE