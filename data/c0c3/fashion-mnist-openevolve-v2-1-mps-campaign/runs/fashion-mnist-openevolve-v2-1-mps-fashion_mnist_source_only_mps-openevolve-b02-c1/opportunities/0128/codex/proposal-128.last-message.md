MECHANISM: Cartesian marginal-preserving translation augmentation

HYPOTHESIS: Training on diagonal as well as cardinal one-pixel translations, while preserving each axis’s existing displacement probabilities, will exceed 9,311 correct validation predictions.

INTENDED_EDIT: Sample vertical and horizontal offsets jointly from a 6×6 Cartesian distribution using one random draw, adding diagonal translations without changing marginal augmentation strength, parameters, or meaningful computation.

EVIDENCE: Weighted transformed-view aggregation reached the best 9,311 correct, showing translation handling affects borderline predictions; the prior independent-axis attempt timed out and produced no contradictory metric evidence.

<<<<<<< SEARCH
    choices = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choices == 2, 0, offset_y)
    offset_y = torch.where(choices == 3, 2, offset_y)
    offset_x = torch.where(choices == 4, 0, offset_x)
    offset_x = torch.where(choices == 5, 2, offset_x)
=======
    choices = torch.randint(0, 36, (batch,), device=images.device)
    choice_y = choices.remainder(6)
    choice_x = torch.div(choices, 6, rounding_mode="floor")
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choice_y == 4, 0, offset_y)
    offset_y = torch.where(choice_y == 5, 2, offset_y)
    offset_x = torch.where(choice_x == 4, 0, offset_x)
    offset_x = torch.where(choice_x == 5, 2, offset_x)
>>>>>>> REPLACE