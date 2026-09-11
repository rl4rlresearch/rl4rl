MECHANISM: Half-step augmentation-boundary refinement

HYPOTHESIS: Moving the broad-to-cardinal transition from 76/128 to 77/128 of training will exceed 9,172 correct predictions by locating a better boundary between the best 76/128 result and the weaker 78/128 result.

INTENDED_EDIT: Use broad ±2 translations for the first 77/128 of training, followed by center/cardinal one-pixel translations for the remaining 51/128.

EVIDENCE: The 76/128 boundary achieved 9,172 correct versus 9,167 at 78/128, while 74/128 fell to 9,160; 77/128 is the untested half-step on the stronger side of this bracket.

<<<<<<< SEARCH
    if step * 32 < total_steps * 19:
=======
    if step * 128 < total_steps * 77:
>>>>>>> REPLACE