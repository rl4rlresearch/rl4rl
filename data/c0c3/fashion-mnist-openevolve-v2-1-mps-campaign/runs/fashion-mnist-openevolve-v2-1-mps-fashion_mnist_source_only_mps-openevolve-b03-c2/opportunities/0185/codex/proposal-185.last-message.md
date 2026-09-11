MECHANISM: Boundary-seeking arithmetic–geometric TTA fusion

HYPOTHESIS: A 3/32 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.214378804397583.

INTENDED_EDIT: Increase the geometric fusion weight from 1/16 to 3/32, retaining all trained-model, view-weighting, and calibration settings.

EVIDENCE: The 1/16 blend preserved 9,257 correct and improved cross-entropy, whereas 1/8 reduced cross-entropy further but lost one correct; 3/32 is the midpoint that probes the remaining ranking boundary.

<<<<<<< SEARCH
            arithmetic_log_probs + math.log(15.0 / 16.0),
            geometric_log_probs + math.log(1.0 / 16.0),
=======
            arithmetic_log_probs + math.log(29.0 / 32.0),
            geometric_log_probs + math.log(3.0 / 32.0),
>>>>>>> REPLACE