MECHANISM: Earlier transition to clean-image fine-tuning

HYPOTHESIS: Shortening translation augmentation from one-half to seven-sixteenths of training will raise validation_correct above 9,256 by emphasizing clean centered examples after the longer eleven-sixteenths phase reduced it to 9,249.

INTENDED_EDIT: End cyclic translation augmentation one-eighth of the schedule earlier and retain the best verified ensemble temperature for tie-breaking cross-entropy.

EVIDENCE: Extending translation training to eleven-sixteenths reduced validation_correct by seven, motivating a conservative move in the opposite direction; Reference Design 2 verified 1.35605 as the best available baseline temperature.

<<<<<<< SEARCH
        return 1.33 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step < total_steps // 2:
=======
    if step < 7 * total_steps // 16:
>>>>>>> REPLACE