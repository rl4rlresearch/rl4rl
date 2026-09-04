MECHANISM: Half-step refinement on the regularized side of the smoothing optimum

HYPOTHESIS: Ending smoothing at 0.0095 will exceed 9,262 correct predictions by staying near the best 0.009 endpoint while adding less late regularization than the regressed 0.010 endpoint.

INTENDED_EDIT: Change the cosine label-smoothing schedule from 0.020→0.009 to 0.020→0.0095, preserving its initial value and all other training behavior.

EVIDENCE: The 0.009 endpoint achieved the best result at 9,262 correct, while 0.010 achieved 9,258 and the tested 0.0085 half-step achieved 9,257; 0.0095 is the remaining untested half-step directly adjacent to the current optimum.

<<<<<<< SEARCH
    smoothing = 0.009 + 0.0055 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.0095 + 0.00525 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE