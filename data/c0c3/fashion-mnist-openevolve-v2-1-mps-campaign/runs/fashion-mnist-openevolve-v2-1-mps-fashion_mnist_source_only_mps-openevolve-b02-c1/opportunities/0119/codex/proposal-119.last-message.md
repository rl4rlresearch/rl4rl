MECHANISM: Marginal-preserving two-axis translation augmentation

HYPOTHESIS: Independently sampling horizontal and vertical one-pixel shifts while preserving the current per-axis displacement probabilities will exceed 9,311 correct predictions by exposing the model to diagonal translations without increasing augmentation strength or computation materially.

INTENDED_EDIT: Replace mutually exclusive cardinal translations with independent per-axis sampling; each axis remains centered with probability 2/3 and shifted each direction with probability 1/6.

EVIDENCE: Weighted transformed-view aggregation improved the best result to 9,311 correct, showing translation handling affects borderline decisions, while recent architectural additions repeatedly timed out; this tests broader spatial invariance through a computation-neutral training-only change.

<<<<<<< SEARCH
    choices = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choices == 2, 0, offset_y)
    offset_y = torch.where(choices == 3, 2, offset_y)
    offset_x = torch.where(choices == 4, 0, offset_x)
    offset_x = torch.where(choices == 5, 2, offset_x)
=======
    choice_y = torch.randint(0, 6, (batch,), device=images.device)
    choice_x = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choice_y == 0, 0, offset_y)
    offset_y = torch.where(choice_y == 1, 2, offset_y)
    offset_x = torch.where(choice_x == 0, 0, offset_x)
    offset_x = torch.where(choice_x == 1, 2, offset_x)
>>>>>>> REPLACE