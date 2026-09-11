MECHANISM: Full-horizon 0.05-to-zero label-smoothing anneal

HYPOTHESIS: Preserving the baseline’s 0.05 initial smoothing while annealing to hard labels will exceed 9,240 correct predictions by combining stronger early regularization with the beneficial zero-smoothing finish.

INTENDED_EDIT: Change only the cosine label-smoothing schedule from 0.05-to-0.01 to 0.05-to-zero.

EVIDENCE: The verified 0.04-to-zero schedule preserved all 9,240 correct predictions and improved cross-entropy to 0.2169272, while ending smoothing halfway reduced correctness to 9,233; this motivates retaining smoothing across the full training horizon while isolating the benefit of a zero endpoint.

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.025 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE