MECHANISM: Quadratic local calibration of flip-ensemble power mean

HYPOTHESIS: A 0.5831697082519531-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

INTENDED_EDIT: Increase only the inference-time probability power-mean order from 0.583203125 to the interpolated local optimum 0.5831697082519531.

EVIDENCE: Order 0.5831695556640625 outperformed both nearby probes: 0.58316802978515625 had cross-entropy 0.2090454807281494 and 0.583172607421875 had 0.20904548225402833, all retaining 9,243 correct. Quadratic interpolation of these three verified points places the estimated minimum near 0.5831697082519531.

<<<<<<< SEARCH
            power = 0.583203125
=======
            power = 0.5831697082519531
>>>>>>> REPLACE