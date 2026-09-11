MECHANISM: Ultra-fine geometric-fusion interpolation

HYPOTHESIS: An 83665/1048576 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870040893555.

INTENDED_EDIT: Set the arithmetic–geometric probability blend to the midpoint between the best qualified 5229/65536 blend and the adjacent 41833/524288 blend, retaining all other settings.

EVIDENCE: The 5229/65536 blend achieved the best observed cross-entropy, 0.21436870040893555, while 41833/524288 also retained 9,257 correct but slightly worsened cross-entropy; their untested midpoint is the most direct refinement around the apparent optimum.

<<<<<<< SEARCH
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(59.0 / 64.0),
            geometric_log_probs + math.log(5.0 / 64.0),
        )
=======
        aggregate_logits = torch.logaddexp(
            arithmetic_log_probs + math.log(964911.0 / 1048576.0),
            geometric_log_probs + math.log(83665.0 / 1048576.0),
        )
>>>>>>> REPLACE