MECHANISM: Quadratic local calibration of flip-ensemble power mean

HYPOTHESIS: A 0.5831697082519531-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

INTENDED_EDIT: Increase only the inference-time probability power-mean order to the interpolated local optimum.

EVIDENCE: The verified order 0.5831695556640625 outperformed the bracketing probes 0.58316802978515625 and 0.583172607421875 while all retained 9,243 correct; quadratic interpolation estimates the minimum near 0.5831697082519531, whose prior timeout supplied no contradictory performance evidence.

<<<<<<< SEARCH
            power = 0.5831695556640625
=======
            power = 0.5831697082519531
>>>>>>> REPLACE