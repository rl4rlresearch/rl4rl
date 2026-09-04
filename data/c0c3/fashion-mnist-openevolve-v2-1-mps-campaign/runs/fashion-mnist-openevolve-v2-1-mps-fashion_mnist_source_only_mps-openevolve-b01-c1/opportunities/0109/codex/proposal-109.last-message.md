MECHANISM: Evaluation-matched triangular crop sampling

HYPOTHESIS: Sampling central training crops with the same separable 1:2:1 weighting used by validation will exceed 9,257 correct predictions by reducing the train–evaluation mismatch in spatial ensemble supervision.

INTENDED_EDIT: Replace uniform sampling of the nine central crop positions with triangular sampling that selects each axis with probabilities 1/4, 1/2, and 1/4.

EVIDENCE: Center-weighted validation aggregation previously improved correctness, while independently sampled uniform translations regressed to 9,248; this isolates distribution alignment without changing the proven architecture, augmentation range, or compute cost.

<<<<<<< SEARCH
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
=======
    central_offsets = (
        1 + torch.randint(0, 2, (2, 2)).sum(dim=0)
    )
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
>>>>>>> REPLACE