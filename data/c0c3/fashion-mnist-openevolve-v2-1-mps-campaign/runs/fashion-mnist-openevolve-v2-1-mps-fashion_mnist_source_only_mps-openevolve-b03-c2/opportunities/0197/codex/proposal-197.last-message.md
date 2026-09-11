MECHANISM: Ultra-fine geometric-fusion boundary bisection

HYPOTHESIS: A 10459/131072 geometric-consensus blend will preserve exactly 9,257 correct predictions while reducing validation cross-entropy below 0.21436870040893555.

INTENDED_EDIT: Increase geometric fusion from 5229/65536 to 10459/131072, the exact midpoint between the best qualified blend and the nearest accuracy-reducing blend, retaining all other settings.

EVIDENCE: The current 5229/65536 blend retained 9,257 correct at 0.21436870040893555 cross-entropy, while the adjacent tested 2615/32768 blend lost one correct; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
            arithmetic_log_probs + math.log(60307.0 / 65536.0),
            geometric_log_probs + math.log(5229.0 / 65536.0),
=======
            arithmetic_log_probs + math.log(120613.0 / 131072.0),
            geometric_log_probs + math.log(10459.0 / 131072.0),
>>>>>>> REPLACE