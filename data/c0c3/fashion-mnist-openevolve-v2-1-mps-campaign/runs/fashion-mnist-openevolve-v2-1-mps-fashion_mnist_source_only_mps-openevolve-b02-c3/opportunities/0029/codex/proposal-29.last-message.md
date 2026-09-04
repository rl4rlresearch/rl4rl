MECHANISM: Fine-grained post-ensemble temperature calibration

HYPOTHESIS: Sharpening the paired-view posterior ensemble from temperature 0.80 to 0.78 will preserve all 9,315 argmax predictions while reducing validation cross-entropy below 0.1919746.

INTENDED_EDIT: Change only the positive inference temperature from 0.80 to 0.78, leaving training and predicted classes unchanged.

EVIDENCE: Sharpening the same paired-view ensemble from 0.90 to 0.80 preserved 9,315 correct predictions and reduced cross-entropy from 0.1971655 to 0.1919746, motivating a smaller further calibration step.

<<<<<<< SEARCH
        ).log() / 0.8
=======
        ).log() / 0.78
>>>>>>> REPLACE