MECHANISM: Three-point parabolic center-weight calibration

HYPOTHESIS: A center weight of 1.8173396587371826 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.18600157318115235.

INTENDED_EDIT: Set the center-view ensemble weight to the float32-representable minimum estimated from the three nearest completed calibration measurements.

EVIDENCE: The current 1.81732177734375 weight produced lower cross-entropy than both 1.81731414794921875 and 1.8173828125 while all retained 9,360 correct; quadratic interpolation predicts a minimum at 1.8173396587371826, and its prior verification timed out without contrary performance evidence.

<<<<<<< SEARCH
        center_weight = 1.81732177734375
=======
        center_weight = 1.8173396587371826
>>>>>>> REPLACE