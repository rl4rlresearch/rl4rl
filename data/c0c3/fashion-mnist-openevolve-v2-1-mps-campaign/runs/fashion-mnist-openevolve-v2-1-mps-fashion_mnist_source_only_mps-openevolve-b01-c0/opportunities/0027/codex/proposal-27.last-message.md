MECHANISM: Shorter-horizon exponential moving average

HYPOTHESIS: Decreasing EMA decay from 0.99 to 0.985 will exceed 9,290 correct predictions by tracking late-training improvements more closely while retaining useful temporal smoothing.

INTENDED_EDIT: Shorten the EMA’s effective averaging window from roughly 100 to 67 optimizer steps, preserving architecture, augmentation, optimizer schedule, and evaluation views.

EVIDENCE: Increasing EMA decay from 0.99 to 0.995 reduced validation correct from 9,290 to 9,282, directly motivating a controlled move toward a shorter rather than longer averaging horizon.

<<<<<<< SEARCH
            decay = 0.99
=======
            decay = 0.985
>>>>>>> REPLACE