MECHANISM: Linearly recency-weighted late-checkpoint averaging

HYPOTHESIS: Giving later snapshots progressively more weight across the successful final-5% window will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by reducing mismatch with final BatchNorm statistics without discarding the earlier snapshots whose removal hurt accuracy.

INTENDED_EDIT: Replace uniform arithmetic snapshot averaging with a linear recency-weighted average while preserving the averaging window, cadence, architecture, and training procedure.

EVIDENCE: Uniform final-5% averaging achieved 9,252 correct with lower cross-entropy than final-10% averaging, whereas restricting averaging to the final 2.5% reduced accuracy to 9,247; gradual recency weighting provides an intermediate refinement without shortening the successful window.

<<<<<<< SEARCH
                update_weight = 1.0 / (average_count + 1)
=======
                update_weight = 2.0 / (average_count + 2)
>>>>>>> REPLACE