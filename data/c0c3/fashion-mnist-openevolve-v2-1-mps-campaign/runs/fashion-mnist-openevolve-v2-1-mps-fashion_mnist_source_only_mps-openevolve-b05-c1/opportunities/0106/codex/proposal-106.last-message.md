MECHANISM: Interpolated short-horizon parameter EMA

HYPOTHESIS: Lowering the EMA decay cap from 0.98 to 0.9775 will exceed 9,358 correct predictions by reducing evaluation-weight lag while retaining more smoothing than the repeatedly unverified 0.975 setting.

INTENDED_EDIT: Reduce only the parameter EMA decay cap from 0.98 to 0.9775.

EVIDENCE: Lowering the cap from 0.99 to 0.985 improved validation correct from 9,351 to 9,357, and 0.98 improved it again to 9,358 with lower cross-entropy; 0.9775 conservatively continues that direction while interpolating between the best verified setting and the twice-timed-out 0.975 candidate.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.9775, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE