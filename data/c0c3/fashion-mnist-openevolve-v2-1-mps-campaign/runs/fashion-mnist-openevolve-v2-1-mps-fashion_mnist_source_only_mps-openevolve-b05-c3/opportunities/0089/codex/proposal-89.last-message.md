MECHANISM: Lower-side BatchNorm EMA refinement

HYPOTHESIS: A 0.0275 terminal EMA rate for BatchNorm buffers will exceed 9,206 correct predictions by testing the unexplored lower side of the best observed 0.03 rate while preserving 0.015 parameter smoothing.

INTENDED_EDIT: Change only the terminal BatchNorm-buffer EMA rate from 0.0325 to 0.0275.

EVIDENCE: A 0.03 buffer rate achieved 9,206 correct with 0.235874 cross-entropy, while 0.0325 tied the correct count but worsened cross-entropy and 0.04 fell to 9,205; 0.0275 is the nearest untested point that brackets 0.03 from below.

<<<<<<< SEARCH
                    ema_rate = 0.0325 if is_buffer else 0.015
=======
                    ema_rate = 0.0275 if is_buffer else 0.015
>>>>>>> REPLACE