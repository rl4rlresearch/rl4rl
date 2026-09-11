MECHANISM: Fine-grained geometric-fusion boundary bisection

HYPOTHESIS: A 163/2048 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436909408569335.

INTENDED_EDIT: Increase geometric fusion from the current 1/64 to 163/2048, the midpoint between the best qualified blend and the nearest tested accuracy-reducing blend.

EVIDENCE: The 81/1024 blend retained 9,257 correct at 0.21436909408569335 cross-entropy, while 41/512 lost one correct; 163/2048 exactly bisects that unresolved boundary.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(63.0 / 64.0),
            geometric_log_probs + math.log(1.0 / 64.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(1885.0 / 2048.0),
            geometric_log_probs + math.log(163.0 / 2048.0),
        )
>>>>>>> REPLACE