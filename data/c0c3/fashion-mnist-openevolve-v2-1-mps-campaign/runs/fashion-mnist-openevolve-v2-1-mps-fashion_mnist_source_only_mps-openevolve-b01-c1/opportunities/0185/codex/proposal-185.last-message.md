MECHANISM: Evaluation-matched triangular crop sampling

HYPOTHESIS: Sampling central training crops with the validation ensemble’s exact 1:2:1 per-axis distribution will exceed 9,267 correct predictions by concentrating updates on the most heavily weighted evaluation views.

INTENDED_EDIT: Replace uniform sampling over the central 3×3 offsets with independent triangular sampling produced by summing two Bernoulli draws per axis.

EVIDENCE: Geometric-blend refinements plateaued at 9,267 correct, while wholesale inference-aligned supervision regressed to 9,237; this surgical change preserves the successful loss and architecture while correcting the remaining central-crop distribution mismatch.

<<<<<<< SEARCH
    central_offsets = torch.randint(1, 4, (2,))
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
=======
    central_offsets = torch.randint(0, 2, (2, 2)).sum(dim=0) + 1
    central_y, central_x = int(central_offsets[0]), int(central_offsets[1])
>>>>>>> REPLACE