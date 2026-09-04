MECHANISM: Hard-target EMA phase

HYPOTHESIS: Removing label smoothing throughout the second half of training will exceed 9,330 correct predictions by eliminating target drift while the classifier EMA is accumulated, while retaining dropout’s proven linear taper.

INTENDED_EDIT: Keep label smoothing at 0.02 before the EMA phase, then switch to hard-label cross-entropy at 50% progress; leave all other training behavior unchanged.

EVIDENCE: Faster classifier EMA improved correctness to 9,330, indicating late classifier lag, while jointly changing the dropout and smoothing taper fell to 9,328; isolating smoothing directly tests whether its moving second-half objective causes that lag.

<<<<<<< SEARCH
    label_smoothing = 0.02 * (1.0 - dropout_decay)
=======
    label_smoothing = 0.02 if progress < 0.50 else 0.0
>>>>>>> REPLACE