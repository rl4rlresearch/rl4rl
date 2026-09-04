MECHANISM: Front-loaded label smoothing with hard-label second half

HYPOTHESIS: Ending label smoothing halfway through training will exceed 9,240 correct predictions by preserving early regularization while allowing a full epoch of hard-label optimization.

INTENDED_EDIT: Replace the cosine 0.04-to-zero smoothing schedule with a linear 0.04-to-zero anneal over the first half of training, followed by zero smoothing.

EVIDENCE: Reducing average smoothing from 0.03 to 0.02 lowered validation cross-entropy from 0.2175833 to 0.2169272 without losing any of the 9,240 correct predictions, motivating a further reduction in smoothing duration while leaving the proven architecture, optimizer, and learning-rate schedule unchanged.

<<<<<<< SEARCH
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.04 * max(1.0 - 2.0 * progress, 0.0)
>>>>>>> REPLACE