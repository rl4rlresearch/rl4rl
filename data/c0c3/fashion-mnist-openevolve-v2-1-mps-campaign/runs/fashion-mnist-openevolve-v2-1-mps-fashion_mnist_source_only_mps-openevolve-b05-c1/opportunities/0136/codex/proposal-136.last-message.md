MECHANISM: Three-point parabolic center-weight calibration

HYPOTHESIS: A center weight of 1.8173396587371826 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.18600157318115235.

INTENDED_EDIT: Replace the center-view ensemble weight with the nearest float32 value to the minimum predicted by quadratic interpolation of the three closest completed measurements.

EVIDENCE: The current 1.81732177734375 weight outperformed both 1.81731414794921875 and 1.8173828125 at the same 9,360 correct; their asymmetric cross-entropy increases predict a local minimum slightly above the current weight.

<<<<<<< SEARCH
        center_weight = 1.81732177734375
=======
        center_weight = 1.8173396587371826
>>>>>>> REPLACE