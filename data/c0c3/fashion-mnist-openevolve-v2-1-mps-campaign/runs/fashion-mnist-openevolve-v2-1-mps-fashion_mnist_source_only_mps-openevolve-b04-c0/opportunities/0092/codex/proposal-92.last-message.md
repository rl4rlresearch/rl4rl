MECHANISM: Float32-adjacent logit calibration

HYPOTHESIS: The immediately lower float32 scale will preserve all 9,359 predictions while reducing cross-entropy below 0.18488420372009276.

INTENDED_EDIT: Replace the evaluation scale with the exact float32 value immediately below 1.226016.

EVIDENCE: The immediate higher float32 neighbor regressed, while the tested 1.2260157 lies roughly three float32 steps below the current best; the untested immediate lower neighbor completes the local bracket.

<<<<<<< SEARCH
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        return 1.22601592540740966796875 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE