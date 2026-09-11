MECHANISM: Conservative post-ensemble temperature sharpening

HYPOTHESIS: Temperature 0.95 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.

INTENDED_EDIT: Divide the unchanged sequential TTA ensemble logits by 0.95 instead of 0.97.

EVIDENCE: Moving from temperature 1.03 to 0.97 improved cross-entropy from 0.197943 to 0.196560 with identical predictions; 0.95 continues that measured direction conservatively, while the more aggressive 0.925 attempts produced only timeouts.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.97
=======
        return (ensemble - math.log(12.0)) / 0.95
>>>>>>> REPLACE