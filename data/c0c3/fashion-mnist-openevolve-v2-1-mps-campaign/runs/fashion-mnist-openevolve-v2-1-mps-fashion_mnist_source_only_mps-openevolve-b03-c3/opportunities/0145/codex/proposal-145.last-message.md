MECHANISM: Uniform radius-two TTA with retained near-shift asymmetry

HYPOTHESIS: Uniformly weighting radius-two translations at 0.00625 while preserving the 9,284-correct model’s vertical/horizontal radius-one asymmetry will exceed 9,284 correct predictions by matching the training distribution for weak outer shifts without removing the apparently useful near-shift directional bias.

INTENDED_EDIT: Replace distance-dependent radius-two logit weights with equal 0.00625 weights; leave architecture, training, radius-one TTA, and total ensemble weight unchanged.

EVIDENCE: The current asymmetric, distance-weighted TTA reached 9,284 correct; symmetrizing radius-one alone reached 9,283, while symmetrizing radius-one and flattening radius-two also reached 9,283. Their equal correct counts motivate isolating outer-weight flattening while retaining the best design’s near-shift asymmetry.

<<<<<<< SEARCH
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
                logits = logits + weight * self._flip_average(shifted)
=======
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
>>>>>>> REPLACE