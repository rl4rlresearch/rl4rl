MECHANISM: Conservative center-weighted ensemble interpolation

HYPOTHESIS: A 15% central-range correction will retain at least 9,063 correct predictions while lowering validation cross-entropy below 0.272932.

INTENDED_EDIT: Increase the central 18-view ensemble weight from 10% to 15%, retaining 85% of the full 50-view ensemble.

EVIDENCE: A 10% correction preserved 9,063 correct and improved cross-entropy from 0.273786 to 0.272932, while 25% lowered cross-entropy further but lost one correct prediction; 15% probes the interval conservatively.

<<<<<<< SEARCH
        return 0.9 * full_ensemble + 0.1 * central_ensemble
=======
        return 0.85 * full_ensemble + 0.15 * central_ensemble
>>>>>>> REPLACE