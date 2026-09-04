MECHANISM: Binomial center-weighted crop aggregation

HYPOTHESIS: Concentrating the proven 10% central-crop allocation toward less-displaced views will exceed 9,247 correct predictions while preserving the locally optimal total central weight.

INTENDED_EDIT: Weight the central 3×3 validation ensemble by a separable 1:2:1 kernel and sample central training offsets from the matching binomial distribution.

EVIDENCE: Changing total central allocation from 10% to either 12.5% or 7.5% regressed to 9,239 and 9,238 correct; preserving 10% while refining its spatial distribution isolates whether central-view quality, rather than total allocation, can improve the ensemble.

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
                    central_weight = (
                        (1, 2, 1)[offset_y - 1]
                        * (1, 2, 1)[offset_x - 1]
                    )
                    if central_logits_sum is None:
                        central_logits_sum = central_weight * view_logits
                    else:
                        central_logits_sum = (
                            central_logits_sum
                            + central_weight * view_logits
                        )
        full_ensemble = logits_sum / 50.0
        central_ensemble = central_logits_sum / 32.0
>>>>>>> REPLACE

<<<<<<< SEARCH
    central_offsets = torch.randint(1, 4, (2,))
=======
    central_offsets = (
        torch.randint(0, 2, (2, 2)).sum(dim=0) + 1
    )
>>>>>>> REPLACE