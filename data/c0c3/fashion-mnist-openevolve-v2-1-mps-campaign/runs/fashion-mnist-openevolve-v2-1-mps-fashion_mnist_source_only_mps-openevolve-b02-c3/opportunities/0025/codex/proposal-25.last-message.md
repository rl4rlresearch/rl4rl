MECHANISM: Stronger post-ensemble temperature calibration

HYPOTHESIS: Sharpening the paired-view posterior ensemble from temperature 0.9 to 0.8 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1971655.

INTENDED_EDIT: Change only the positive inference temperature from 0.9 to 0.8, leaving training and prediction classes unchanged.

EVIDENCE: Temperature 0.9 preserved 9,274 correct predictions while reducing cross-entropy from 0.2166819 to 0.2070246; the current paired-view model retains the same probability-ensemble mechanism and has reached 9,315 correct.

<<<<<<< SEARCH
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.9
=======
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.8
>>>>>>> REPLACE