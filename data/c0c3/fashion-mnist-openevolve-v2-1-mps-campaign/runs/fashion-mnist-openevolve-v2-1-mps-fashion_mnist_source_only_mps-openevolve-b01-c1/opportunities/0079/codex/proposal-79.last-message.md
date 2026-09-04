MECHANISM: Moderately center-weighted crop aggregation

HYPOTHESIS: A separable 2:3:2 central-crop distribution will exceed 9,247 correct predictions by capturing part of the binomial ensemble’s cross-entropy improvement without its excessive center concentration.

INTENDED_EDIT: Sample central training offsets and weight central validation crops with a separable 2:3:2 kernel, preserving the proven 10% central allocation.

EVIDENCE: The stronger 1:2:1 kernel improved cross-entropy from 0.21609 to 0.21532 but reduced correctness by one; an intermediate kernel directly tests whether weaker concentration preserves baseline decisions while gaining better-calibrated ones.

<<<<<<< SEARCH
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    if central_logits_sum is None:
                        central_logits_sum = view_logits
                    else:
                        central_logits_sum = central_logits_sum + view_logits
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 18.0
=======
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    central_weight_y = 3.0 if offset_y == 2 else 2.0
                    central_weight_x = 3.0 if offset_x == 2 else 2.0
                    weighted_logits = (
                        central_weight_y * central_weight_x * view_logits
                    )
                    if central_logits_sum is None:
                        central_logits_sum = weighted_logits
                    else:
                        central_logits_sum = (
                            central_logits_sum + weighted_logits
                        )
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 98.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
=======
    central_draws = torch.randint(0, 7, (2,))
    central_y = (
        1
        + int(central_draws[0] >= 2)
        + int(central_draws[0] >= 5)
    )
    central_x = (
        1
        + int(central_draws[1] >= 2)
        + int(central_draws[1] >= 5)
    )
>>>>>>> REPLACE