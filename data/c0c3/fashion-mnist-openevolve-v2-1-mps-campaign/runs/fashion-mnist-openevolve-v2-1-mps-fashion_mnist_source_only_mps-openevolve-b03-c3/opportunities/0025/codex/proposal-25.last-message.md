MECHANISM: Wider cosine target-sharpening curriculum

HYPOTHESIS: Widening label smoothing from 0.06 early to 0.00 late, while preserving its 0.03 average, will exceed 9,240 correct predictions by strengthening early regularization and optimizing hard-label margins near convergence.

INTENDED_EDIT: Increase the cosine label-smoothing amplitude from 0.02 to 0.03, changing the schedule from 0.05→0.01 to 0.06→0.00.

EVIDENCE: The current 0.05→0.01 curriculum improved Reference Design 3 from 9,237 to 9,240 correct and reduced cross-entropy from 0.24180 to 0.23423; widening the same successful curriculum directly tests whether stronger early smoothing and complete late sharpening extend that gain.

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.03 + 0.03 * math.cos(math.pi * progress)
>>>>>>> REPLACE