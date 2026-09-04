MECHANISM: One-dimensional ensemble temperature calibration

HYPOTHESIS: Temperature 0.934 will preserve exactly 9,311 argmax predictions while reducing validation cross-entropy below 0.1922903751.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.930 to 0.934.

EVIDENCE: Temperatures 0.900, 0.912, 0.924, and 0.930 progressively reduced cross-entropy without changing the 9,311 correct predictions; the measured curve places its minimum near 0.934, while prior nearby attempts timed out without contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.930)
=======
        return ensemble / (12.0 * 0.934)
>>>>>>> REPLACE