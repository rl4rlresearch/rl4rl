MECHANISM: Isolated post-ensemble calibration sharpening

HYPOTHESIS: Scaling the unchanged ten-view ensemble by 1.20 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2235582.

INTENDED_EDIT: Increase only the positive inference-time ensemble scale from 1.10 to 1.20.

EVIDENCE: Raising the scale from 1.05 to 1.10 preserved all 9,252 predictions and reduced cross-entropy from 0.2287977 to 0.2235582; prior 1.20 trials also changed evaluation mechanics, so this isolates calibration on the only verified-completing path.

<<<<<<< SEARCH
        return 1.10 * ensemble_log_probabilities
=======
        return 1.20 * ensemble_log_probabilities
>>>>>>> REPLACE