MECHANISM: Further extended inference-aligned augmentation curriculum

HYPOTHESIS: Moving the broad-to-cardinal transition from 38/64 to 37/64 of training will exceed 9,172 correct predictions by continuing the observed monotonic improvement from successively longer terminal inference-aligned phases.

INTENDED_EDIT: Use broad ±2 translations for the first 37/64 of training, then center/cardinal one-pixel translations for the remaining 27/64.

EVIDENCE: Extending the terminal phase from 25/64 to 26/64 raised validation_correct from 9,167 to 9,172, continuing the earlier monotonic gains at terminal phases of one quarter, one third, and three eighths; the next one-step boundary change is the most informative continuation.

<<<<<<< SEARCH
    if step * 32 < total_steps * 19:
=======
    if step * 64 < total_steps * 37:
>>>>>>> REPLACE