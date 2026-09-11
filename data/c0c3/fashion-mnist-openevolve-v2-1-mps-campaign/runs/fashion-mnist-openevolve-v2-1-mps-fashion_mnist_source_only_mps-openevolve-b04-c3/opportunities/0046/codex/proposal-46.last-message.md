MECHANISM: Interpolated terminal-rate cosine decay

HYPOTHESIS: A 10.5% terminal learning-rate floor will exceed 9,240 correct predictions by targeting the estimated optimum between the improving 7.5%-to-12.5% interval and the sharply weaker 15% result.

INTENDED_EDIT: Change only the cosine schedule’s terminal floor from 7.5% to 10.5%, preserving the verified architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.

EVIDENCE: The identical model achieved 9,233 correct at 7.5%, 9,240 at 12.5%, and 9,219 at 15%; these results bracket a narrow optimum below 12.5%, motivating a lower interpolation without adding runtime or parameters.

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.105 + 0.895 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE