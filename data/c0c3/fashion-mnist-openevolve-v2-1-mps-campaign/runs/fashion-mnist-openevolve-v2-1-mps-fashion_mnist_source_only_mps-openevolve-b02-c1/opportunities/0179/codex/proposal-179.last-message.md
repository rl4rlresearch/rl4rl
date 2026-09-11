MECHANISM: Train–test geometric-view distribution alignment

HYPOTHESIS: Sampling centered training views 2.25 times as often as each shifted view will exceed 9,311 correct predictions by aligning learned invariances with the verified center-emphasized TTA ensemble.

INTENDED_EDIT: Change augmentation sampling from a 2:1 center-to-each-shift ratio to the TTA-matched 2.25:1 ratio, using probabilities 9/25 for centered views and 4/25 for each one-pixel shift.

EVIDENCE: Raising evaluation center weights from 2.0 to 2.25 preserved 9,311 correct while lowering cross-entropy to 0.1922469223, indicating a modest center emphasis is beneficial; applying that same ratio during learning can change decision boundaries without adding parameters or runtime.

<<<<<<< SEARCH
    choices = torch.randint(0, 6, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(choices == 2, 0, offset_y)
    offset_y = torch.where(choices == 3, 2, offset_y)
    offset_x = torch.where(choices == 4, 0, offset_x)
    offset_x = torch.where(choices == 5, 2, offset_x)
=======
    choices = torch.randint(0, 25, (batch,), device=images.device)
    offset_y = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_x = torch.ones(batch, dtype=torch.long, device=images.device)
    offset_y = torch.where(
        (choices >= 9) & (choices < 13), 0, offset_y
    )
    offset_y = torch.where(
        (choices >= 13) & (choices < 17), 2, offset_y
    )
    offset_x = torch.where(
        (choices >= 17) & (choices < 21), 0, offset_x
    )
    offset_x = torch.where(choices >= 21, 2, offset_x)
>>>>>>> REPLACE