MECHANISM: Training-to-ensemble view-frequency matching

HYPOTHESIS: Extending shift augmentation through 20/29 of training will raise validation_correct above 9,256 by matching training view frequencies to the verified 3.25:1 evaluation ensemble.

INTENDED_EDIT: Change only the duration of deterministic shift augmentation from one-half to 20/29 of optimizer steps.

EVIDENCE: The verified ensemble assigns center and four shifted views normalized weights of 13/29 and 4/29 each. Augmenting for 20/29 of training produces those same marginal frequencies, while preserving the established per-view loss because the probability-ensemble loss reduced validation_correct to 9,248.

<<<<<<< SEARCH
    if step < total_steps // 2:
=======
    if step < (20 * total_steps) // 29:
>>>>>>> REPLACE