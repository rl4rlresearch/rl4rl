MECHANISM: Stronger tail-weight averaging

HYPOTHESIS: Increasing the final interpolation toward the averaged tail weights from 0.5 to 0.75 will change learned decision boundaries and exceed 9,311 correct predictions without increasing runtime or parameters.

INTENDED_EDIT: Apply 75% rather than 50% of the existing late-training parameter average at the final optimizer step.

EVIDENCE: Evaluation-temperature and TTA-weight tuning plateaued at 9,311 correct, while architectural changes either reduced accuracy or timed out; strengthening the already-computed tail average is a low-cost way to improve learned generalization.

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.75)
>>>>>>> REPLACE