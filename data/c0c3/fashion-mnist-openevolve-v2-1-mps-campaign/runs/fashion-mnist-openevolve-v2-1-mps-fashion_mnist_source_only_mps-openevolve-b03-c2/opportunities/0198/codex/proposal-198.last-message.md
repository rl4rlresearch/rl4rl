MECHANISM: Terminal geometric-fusion boundary bisection

HYPOTHESIS: A 20917/262144 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870040893555.

INTENDED_EDIT: Increase geometric fusion to the sole representable midpoint between the best qualified 5229/65536 blend and the accuracy-reducing 10459/131072 blend, retaining all other settings.

EVIDENCE: The 5229/65536 blend retained 9,257 correct at 0.21436870040893555 cross-entropy, while 10459/131072 lost one correct; 20917/262144 exactly bisects their final unresolved interval.

<<<<<<< SEARCH
            arithmetic_log_probs + math.log(15077.0 / 16384.0),
            geometric_log_probs + math.log(1307.0 / 16384.0),
=======
            arithmetic_log_probs + math.log(241227.0 / 262144.0),
            geometric_log_probs + math.log(20917.0 / 262144.0),
>>>>>>> REPLACE