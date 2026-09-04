MECHANISM: Terminal clean-loss consolidation

HYPOTHESIS: Completing the proven dropout and label-smoothing taper at 90% progress will exceed 9,330 correct predictions by providing a fully unregularized final optimization window before EMA evaluation.

INTENDED_EDIT: Preserve regularization through the first half of training, then taper it to zero over the next 40% instead of retaining it until the final step.

EVIDENCE: Classifier EMA peaked at a faster 0.04 rate, while two-mask dropout consistency fell to 9,307, indicating that late classifier adaptation matters but additional dropout regularization is harmful; a clean terminal window targets that adaptation without changing the established architecture, optimizer, or EMA.

<<<<<<< SEARCH
    dropout_decay = max(progress - 0.50, 0.0) / 0.50
=======
    dropout_decay = min(
        max(progress - 0.50, 0.0) / 0.40,
        1.0,
    )
>>>>>>> REPLACE