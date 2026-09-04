MECHANISM: Evidence-fitted post-ensemble temperature sharpening

HYPOTHESIS: Temperature 0.925 will preserve all 9,286 argmax predictions while reducing validation cross-entropy below 0.196560.

INTENDED_EDIT: Sharpen the unchanged 12-view ensemble by changing its evaluation temperature from 0.97 to 0.925.

EVIDENCE: Temperature 1.03 worsened cross-entropy to 0.197943, while 0.97 improved it to 0.196560 with identical predictions; this directly supports further sharpening, and previous 0.925 attempts timed out without producing contrary validation evidence.

<<<<<<< SEARCH
        return (ensemble - math.log(12.0)) / 0.97
=======
        return (ensemble - math.log(12.0)) / 0.925
>>>>>>> REPLACE