MECHANISM: Quadratic-refined crop power fusion

HYPOTHESIS: Setting the crop fusion power to 1.3477 will preserve 9,349 correct predictions while lowering validation cross-entropy below 0.1876555618286133.

INTENDED_EDIT: Move the fusion power from 1.34 to the local cross-entropy minimum estimated from the three most relevant measured powers.

EVIDENCE: Cross-entropy decreased from 0.1876840835571289 at 1.087 to 0.18766820755004882 at 1.174 and 0.1876555618286133 at 1.34; a quadratic fit to these nonuniformly spaced observations places the minimum near 1.3477, while the small adjustment should retain the additional correct prediction gained at 1.34.

<<<<<<< SEARCH
        fusion_power = 1.34
=======
        fusion_power = 1.3477
>>>>>>> REPLACE